//! Per-device GPU facts used by runtime admission policy.

use crate::error::{QueryError, Result};

/// Snapshot of the stable facts nexus needs for one CUDA device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    pub device: usize,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub sm_count: Option<u32>,
    pub utilization_pct: Option<u8>,
}

/// Query GPU facts behind a fakeable interface.
pub trait GpuInventory: Send + Sync {
    fn device_count(&self) -> Result<usize>;

    fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts>;

    fn all_facts(&self) -> Result<Vec<DeviceFacts>> {
        let device_count = self.device_count()?;
        let mut facts = Vec::with_capacity(device_count);
        for device_ordinal in 0..device_count {
            facts.push(self.facts(device_ordinal)?);
        }
        Ok(facts)
    }
}

/// Production GPU inventory backed by cudf-rs.
#[derive(Debug, Default, Clone, Copy)]
pub struct CudfInventory;

impl GpuInventory for CudfInventory {
    fn device_count(&self) -> Result<usize> {
        cudf::DeviceId::count().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_inventory.device_count",
                "failed to query CUDA device count",
                error,
            )
        })
    }

    fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
        let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_inventory.device_id",
                "failed to resolve CUDA device id while collecting inventory facts",
                error,
            )
            .with_fact("device_ordinal", device_ordinal)
        })?;
        let memory = device.available_memory().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_inventory.memory",
                "failed to query CUDA device memory inventory",
                error,
            )
            .with_fact("device_ordinal", device.ordinal())
        })?;
        let sm_count = device.multiprocessor_count().ok().filter(|count| *count > 0);
        Ok(DeviceFacts {
            device: device_ordinal,
            total_bytes: memory.total_bytes,
            free_bytes: memory.free_bytes,
            sm_count,
            utilization_pct: utilization_pct(device),
        })
    }
}

/// Convert stable device capacity facts into a bounded query budget.
pub fn device_budget_bytes_from_facts(facts: &DeviceFacts, headroom_bytes: u64, fallback_bytes: u64) -> u64 {
    facts.total_bytes.checked_sub(headroom_bytes).filter(|value| *value > 0).unwrap_or(fallback_bytes)
}

#[cfg(feature = "nvml")]
fn utilization_pct(device: cudf::DeviceId) -> Option<u8> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let pci_bus_id = device.pci_bus_id().ok()?;
    let rates = nvml.device_by_pci_bus_id(pci_bus_id).ok()?.utilization_rates().ok()?;
    u8::try_from(rates.gpu).ok()
}

#[cfg(not(feature = "nvml"))]
fn utilization_pct(_device: cudf::DeviceId) -> Option<u8> {
    None
}
