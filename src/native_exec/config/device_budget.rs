use super::*;

impl NativeCudfExecutionConfig {
    /// Returns the auto-sized bounded device budget in bytes.
    ///
    /// The value is derived from the primary GPU's reported total memory minus
    /// 500 MiB of libcudf/driver headroom. A 256 MiB fallback keeps bounded
    /// report and service defaults constructible on hosts without CUDA
    /// inventory or without a visible GPU.
    pub fn auto_sized_device_budget_bytes() -> u64 {
        static AUTO_SIZED_DEVICE_BUDGET_BYTES: OnceLock<u64> = OnceLock::new();
        *AUTO_SIZED_DEVICE_BUDGET_BYTES.get_or_init(detect_auto_sized_device_budget_bytes)
    }

    /// Returns the auto-sized bounded device budget for `device_ordinal`.
    ///
    /// Unlike [`Self::auto_sized_device_budget_bytes`], this helper is not
    /// process-cached because server placement can choose a non-default ordinal.
    pub fn auto_sized_device_budget_bytes_for_device(device_ordinal: usize) -> u64 {
        auto_sized_device_budget_bytes_for_device_with_inventory(&CudfInventory, device_ordinal)
    }

    /// Returns the auto-sized bounded device budget for `device_ordinal` after
    /// applying a total-memory budget fraction.
    ///
    /// The fraction is an upper bound on usable device memory. The fixed
    /// headroom remains a floor, so `1.0` preserves the historical
    /// `total - 500 MiB` behavior instead of granting the full device.
    pub fn auto_sized_device_budget_bytes_for_device_with_fraction(device_ordinal: usize, budget_fraction: f64) -> u64 {
        auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(
            &CudfInventory,
            device_ordinal,
            budget_fraction,
        )
    }
}

pub(super) fn detect_auto_sized_device_budget_bytes() -> u64 {
    auto_sized_device_budget_bytes_with_inventory(&CudfInventory)
}

pub(super) fn auto_sized_device_budget_bytes_with_inventory(inventory: &impl GpuInventory) -> u64 {
    auto_sized_device_budget_bytes_for_device_with_inventory(inventory, cudf::DeviceId::DEFAULT.ordinal())
}

pub(super) fn auto_sized_device_budget_bytes_for_device_with_inventory(
    inventory: &impl GpuInventory,
    device_ordinal: usize,
) -> u64 {
    inventory
        .facts(device_ordinal)
        .map(|facts| {
            device_budget_bytes_from_facts(
                &facts,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            )
        })
        .unwrap_or(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES)
}

pub(super) fn auto_sized_device_budget_bytes_for_device_with_fraction_and_inventory(
    inventory: &impl GpuInventory,
    device_ordinal: usize,
    budget_fraction: f64,
) -> u64 {
    inventory
        .facts(device_ordinal)
        .map(|facts| {
            let headroom_budget = device_budget_bytes_from_facts(
                &facts,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES,
                NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            );
            let fraction_budget = fraction_budget_bytes(facts.total_bytes, budget_fraction)
                .filter(|budget| *budget > 0)
                .unwrap_or(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES);
            headroom_budget.min(fraction_budget)
        })
        .unwrap_or(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES)
}

pub(super) fn fraction_budget_bytes(total_bytes: u64, budget_fraction: f64) -> Option<u64> {
    if !budget_fraction.is_finite() || !(0.0..=1.0).contains(&budget_fraction) || budget_fraction == 0.0 {
        return None;
    }
    let budget = (total_bytes as f64 * budget_fraction).floor();
    (budget > 0.0 && budget <= u64::MAX as f64).then_some(budget as u64)
}
