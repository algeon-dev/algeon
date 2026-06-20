//! Placement policies for independent query scheduling across devices.

use super::DeviceRegistry;
use crate::error::Result;

/// Placement input for one admitted query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementRequest {
    /// Minimum device-memory bytes required to place the query.
    pub min_device_budget_bytes: u64,
    /// Preferred device-memory bytes for placement.
    pub want_device_budget_bytes: u64,
    /// Optional tenant identifier for affinity policies.
    pub tenant_id: Option<String>,
}

impl PlacementRequest {
    /// Creates a placement request for a fixed query budget.
    pub fn new(device_budget_bytes: u64) -> Self {
        Self {
            min_device_budget_bytes: device_budget_bytes,
            want_device_budget_bytes: device_budget_bytes,
            tenant_id: None,
        }
    }

    /// Creates a placement request with separate minimum and preferred budgets.
    pub fn with_grant_range(min_device_budget_bytes: u64, want_device_budget_bytes: u64) -> Self {
        Self { min_device_budget_bytes, want_device_budget_bytes, tenant_id: None }
    }

    /// Adds tenant identity for policies that need tenant affinity.
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }
}

/// Policy that returns ordered candidate devices, most preferred first.
pub trait PlacementPolicy: Send + Sync {
    /// Ordered placement candidates for `request`.
    fn candidates(&self, registry: &DeviceRegistry, request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>>;
}

/// Places queries on the device with most ungranted capacity, then fewer actives.
#[derive(Debug, Default, Clone, Copy)]
pub struct LeastLoadedPlacement;

impl PlacementPolicy for LeastLoadedPlacement {
    fn candidates(&self, registry: &DeviceRegistry, _request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>> {
        let mut loads = Vec::with_capacity(registry.len());
        for context in registry.contexts() {
            loads.push(DeviceLoad {
                ordinal: context.device().ordinal(),
                available_bytes: context.available_bytes()?,
                active_queries: context.active_queries(),
            });
        }
        Ok(rank_device_loads(loads)
            .into_iter()
            .filter_map(|ordinal| registry.context_by_ordinal(ordinal).ok().map(|context| context.device()))
            .collect())
    }
}

/// Pins all queries to one device when that device is part of the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedDevicePlacement {
    pub device: cudf::DeviceId,
}

impl PlacementPolicy for PinnedDevicePlacement {
    fn candidates(&self, registry: &DeviceRegistry, _request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>> {
        Ok(registry.context(self.device).map(|_| vec![self.device]).unwrap_or_default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeviceLoad {
    ordinal: usize,
    available_bytes: u64,
    active_queries: usize,
}

fn rank_device_loads(mut loads: Vec<DeviceLoad>) -> Vec<usize> {
    loads.sort_by(|left, right| {
        right
            .available_bytes
            .cmp(&left.available_bytes)
            .then_with(|| left.active_queries.cmp(&right.active_queries))
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });
    loads.into_iter().map(|load| load.ordinal).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_loaded_ranking_prefers_available_bytes_then_active_count_then_ordinal() {
        let ranked = rank_device_loads(vec![
            DeviceLoad { ordinal: 2, available_bytes: 10, active_queries: 0 },
            DeviceLoad { ordinal: 1, available_bytes: 20, active_queries: 2 },
            DeviceLoad { ordinal: 0, available_bytes: 20, active_queries: 1 },
            DeviceLoad { ordinal: 3, available_bytes: 20, active_queries: 1 },
        ]);

        assert_eq!(ranked, vec![0, 3, 1, 2]);
    }
}
