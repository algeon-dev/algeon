//! Per-device GPU facts and live diagnostics telemetry.

#[cfg(feature = "nvml")]
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use query_planning::error::Result;
use serde::Serialize;

use crate::native_error::cudf_query_error;

/// Availability of one live device-telemetry substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum DeviceTelemetryStatus {
    /// The substrate returned a live sample.
    Available,
    /// The optional telemetry feature was not compiled.
    FeatureDisabled,
    /// The substrate or device sample was unavailable.
    Unavailable,
}

impl DeviceTelemetryStatus {
    /// Stable diagnostics label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::FeatureDisabled => "feature_disabled",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Live CUDA and NVML telemetry for one device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceTelemetrySnapshot {
    /// Wall-clock sample time for correlating diagnostics with external telemetry.
    pub(crate) sampled_at_unix_ms: u64,
    pub device_ordinal: usize,
    pub cuda_status: DeviceTelemetryStatus,
    pub(crate) cuda_total_bytes: Option<u64>,
    pub cuda_free_bytes: Option<u64>,
    pub nvml_status: DeviceTelemetryStatus,
    pub(crate) nvml_total_bytes: Option<u64>,
    pub(crate) nvml_reserved_bytes: Option<u64>,
    pub nvml_used_bytes: Option<u64>,
    pub(crate) nvml_free_bytes: Option<u64>,
    pub gpu_utilization_pct: Option<u32>,
    pub memory_utilization_pct: Option<u32>,
}

/// Fakeable provider for live device telemetry.
///
/// Telemetry is diagnostics only; it is not a substitute data source for the
/// admission ledger.
pub(crate) trait GpuTelemetry: Send + Sync {
    /// Samples live telemetry for one device.
    ///
    /// Sampling does not fail. An unavailable substrate, a disabled telemetry
    /// feature, or an unresolvable ordinal is reported through the snapshot's
    /// status fields with the affected values left `None`, and the CUDA and
    /// NVML substrates can be unavailable independently.
    fn snapshot(&self, device_ordinal: usize) -> DeviceTelemetrySnapshot;
}

pub(crate) fn device_telemetry_snapshots_for_ordinals(
    telemetry: &dyn GpuTelemetry,
    device_ordinals: &[usize],
) -> Vec<DeviceTelemetrySnapshot> {
    device_ordinals.iter().map(|ordinal| telemetry.snapshot(*ordinal)).collect()
}

/// Production live telemetry backed by CUDA plus optional NVML.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct AlgeonNvmlTelemetry;

impl GpuTelemetry for AlgeonNvmlTelemetry {
    fn snapshot(&self, device_ordinal: usize) -> DeviceTelemetrySnapshot {
        let mut snapshot = DeviceTelemetrySnapshot {
            sampled_at_unix_ms: sampled_at_unix_ms(),
            device_ordinal,
            cuda_status: DeviceTelemetryStatus::Unavailable,
            cuda_total_bytes: None,
            cuda_free_bytes: None,
            nvml_status: nvml_status_without_sample(),
            nvml_total_bytes: None,
            nvml_reserved_bytes: None,
            nvml_used_bytes: None,
            nvml_free_bytes: None,
            gpu_utilization_pct: None,
            memory_utilization_pct: None,
        };
        let Ok(device) = cudf::DeviceId::new(device_ordinal) else {
            return snapshot;
        };
        if let Ok(memory) = device.available_memory() {
            snapshot.cuda_status = DeviceTelemetryStatus::Available;
            snapshot.cuda_total_bytes = Some(memory.total_bytes);
            snapshot.cuda_free_bytes = Some(memory.free_bytes);
        }
        sample_nvml(device, &mut snapshot);
        snapshot
    }
}

fn sampled_at_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

/// Snapshot of the stable facts algeon needs for one CUDA device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    pub device: usize,
    /// NVML product name, when the optional NVML diagnostics substrate is available.
    pub model_name: Option<String>,
    /// NVML UUID, when the optional NVML diagnostics substrate is available.
    pub uuid: Option<String>,
    /// NVML-reported physical framebuffer capacity, when the optional NVML diagnostics substrate is available.
    pub physical_total_bytes: Option<u64>,
    /// CUDA-visible allocation capacity. This can exclude NVML-reserved framebuffer capacity.
    pub total_bytes: u64,
    /// Free bytes at sampling time. The registry's post-install capacity
    /// freeze reads this once per device; it is never consulted during
    /// admission.
    pub(crate) free_bytes: u64,
    pub(crate) sm_count: Option<u32>,
    pub(crate) utilization_pct: Option<u8>,
}

/// Query GPU facts behind a fakeable interface.
pub(crate) trait GpuInventory: Send + Sync {
    /// Facts for one device ordinal.
    ///
    /// `Err` means the required CUDA facts for the ordinal are unavailable.
    /// Optional fields are `None` when their diagnostics substrate is
    /// unavailable rather than failing the query.
    fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts>;
}

/// Production GPU inventory backed by cudf.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct AlgeonGpuInventory;

impl GpuInventory for AlgeonGpuInventory {
    fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
        let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
            cudf_query_error(
                "query_engine.gpu_inventory.device_id",
                "failed to resolve CUDA device id while collecting inventory facts",
                error,
            )
            .with_fact("device_ordinal", device_ordinal)
        })?;
        let memory = device.available_memory().map_err(|error| {
            cudf_query_error("query_engine.gpu_inventory.memory", "failed to query CUDA device memory inventory", error)
                .with_fact("device_ordinal", device.ordinal())
        })?;
        let sm_count = device.multiprocessor_count().ok().filter(|count| *count > 0);
        let (model_name, uuid, physical_total_bytes) = nvml_device_identity(device);
        Ok(DeviceFacts {
            device: device_ordinal,
            model_name,
            uuid,
            physical_total_bytes,
            total_bytes: memory.total_bytes,
            free_bytes: memory.free_bytes,
            sm_count,
            utilization_pct: utilization_pct(device),
        })
    }
}

#[cfg(feature = "nvml")]
fn nvml_device_identity(device: cudf::DeviceId) -> (Option<String>, Option<String>, Option<u64>) {
    let Some(nvml) = shared_nvml() else {
        return (None, None, None);
    };
    let Ok(pci_bus_id) = device.pci_bus_id() else {
        return (None, None, None);
    };
    let Ok(device) = nvml.device_by_pci_bus_id(pci_bus_id) else {
        return (None, None, None);
    };
    (device.name().ok(), device.uuid().ok(), device.memory_info().ok().map(|memory| memory.total))
}

#[cfg(not(feature = "nvml"))]
fn nvml_device_identity(_device: cudf::DeviceId) -> (Option<String>, Option<String>, Option<u64>) {
    (None, None, None)
}

/// Convert stable device capacity facts into a bounded query budget.
#[cfg(test)]
pub(crate) fn device_budget_bytes_from_facts(facts: &DeviceFacts, headroom_bytes: u64, fallback_bytes: u64) -> u64 {
    facts.total_bytes.checked_sub(headroom_bytes).filter(|value| *value > 0).unwrap_or(fallback_bytes)
}

#[cfg(feature = "nvml")]
fn utilization_pct(device: cudf::DeviceId) -> Option<u8> {
    let nvml = shared_nvml()?;
    let pci_bus_id = device.pci_bus_id().ok()?;
    let rates = nvml.device_by_pci_bus_id(pci_bus_id).ok()?.utilization_rates().ok()?;
    u8::try_from(rates.gpu).ok()
}

#[cfg(not(feature = "nvml"))]
fn utilization_pct(_device: cudf::DeviceId) -> Option<u8> {
    None
}

#[cfg(feature = "nvml")]
fn shared_nvml() -> Option<&'static nvml_wrapper::Nvml> {
    static NVML: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();
    NVML.get_or_init(|| nvml_wrapper::Nvml::init().ok()).as_ref()
}

#[cfg(feature = "nvml")]
fn nvml_status_without_sample() -> DeviceTelemetryStatus {
    DeviceTelemetryStatus::Unavailable
}

#[cfg(not(feature = "nvml"))]
fn nvml_status_without_sample() -> DeviceTelemetryStatus {
    DeviceTelemetryStatus::FeatureDisabled
}

#[cfg(feature = "nvml")]
fn sample_nvml(device: cudf::DeviceId, snapshot: &mut DeviceTelemetrySnapshot) {
    let Some(nvml) = shared_nvml() else {
        return;
    };
    let Ok(pci_bus_id) = device.pci_bus_id() else {
        return;
    };
    let Ok(device) = nvml.device_by_pci_bus_id(pci_bus_id) else {
        return;
    };
    let mut sampled = false;
    if let Ok(memory) = device.memory_info() {
        sampled = true;
        snapshot.nvml_total_bytes = Some(memory.total);
        snapshot.nvml_reserved_bytes = Some(memory.reserved);
        snapshot.nvml_used_bytes = Some(memory.used);
        snapshot.nvml_free_bytes = Some(memory.free);
    }
    if let Ok(utilization) = device.utilization_rates() {
        sampled = true;
        snapshot.gpu_utilization_pct = Some(utilization.gpu);
        snapshot.memory_utilization_pct = Some(utilization.memory);
    }
    if sampled {
        snapshot.nvml_status = DeviceTelemetryStatus::Available;
    }
}

#[cfg(not(feature = "nvml"))]
fn sample_nvml(_device: cudf::DeviceId, _snapshot: &mut DeviceTelemetrySnapshot) {}

#[cfg(test)]
mod tests {
    use super::{
        DeviceTelemetrySnapshot, DeviceTelemetryStatus, GpuTelemetry, device_telemetry_snapshots_for_ordinals,
    };

    struct FakeTelemetry;

    impl GpuTelemetry for FakeTelemetry {
        fn snapshot(&self, device_ordinal: usize) -> DeviceTelemetrySnapshot {
            let nvml_available = device_ordinal == 3;
            DeviceTelemetrySnapshot {
                sampled_at_unix_ms: 42,
                device_ordinal,
                cuda_status: DeviceTelemetryStatus::Available,
                cuda_total_bytes: Some(1_000 + device_ordinal as u64),
                cuda_free_bytes: Some(500),
                nvml_status: if nvml_available {
                    DeviceTelemetryStatus::Available
                } else {
                    DeviceTelemetryStatus::Unavailable
                },
                nvml_total_bytes: nvml_available.then_some(1_000),
                nvml_reserved_bytes: None,
                nvml_used_bytes: nvml_available.then_some(250),
                nvml_free_bytes: nvml_available.then_some(750),
                gpu_utilization_pct: nvml_available.then_some(12),
                memory_utilization_pct: nvml_available.then_some(34),
            }
        }
    }

    #[test]
    fn telemetry_provider_samples_every_configured_ordinal_in_order() {
        let snapshots = device_telemetry_snapshots_for_ordinals(&FakeTelemetry, &[3, 1]);

        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].device_ordinal, 3);
        assert_eq!(snapshots[0].cuda_total_bytes, Some(1_003));
        assert_eq!(snapshots[0].nvml_status, DeviceTelemetryStatus::Available);
        assert_eq!(snapshots[0].gpu_utilization_pct, Some(12));
        assert_eq!(snapshots[1].device_ordinal, 1);
        assert_eq!(snapshots[1].nvml_status, DeviceTelemetryStatus::Unavailable);
    }

    #[cfg(feature = "nvml")]
    #[test]
    fn live_telemetry_covers_every_cuda_device_with_valid_ranges() {
        use super::AlgeonNvmlTelemetry;

        let device_count = cudf::DeviceId::count().expect("CUDA device count");
        assert!(device_count > 0, "GPU test lane requires at least one CUDA device");
        let ordinals = (0..device_count).collect::<Vec<_>>();

        let snapshots = device_telemetry_snapshots_for_ordinals(&AlgeonNvmlTelemetry, &ordinals);

        assert_eq!(snapshots.len(), device_count);
        for (ordinal, snapshot) in snapshots.iter().enumerate() {
            assert_eq!(snapshot.device_ordinal, ordinal);
            assert_eq!(snapshot.cuda_status, DeviceTelemetryStatus::Available);
            assert!(snapshot.cuda_free_bytes.unwrap() <= snapshot.cuda_total_bytes.unwrap());
            assert_eq!(snapshot.nvml_status, DeviceTelemetryStatus::Available);
            assert!(snapshot.nvml_free_bytes.unwrap() <= snapshot.nvml_total_bytes.unwrap());
            assert!(snapshot.gpu_utilization_pct.unwrap() <= 100);
            assert!(snapshot.memory_utilization_pct.unwrap() <= 100);
        }
    }
}
