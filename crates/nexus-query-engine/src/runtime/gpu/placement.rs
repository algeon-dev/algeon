//! Placement policies for independent query scheduling across devices.

#[cfg(feature = "cugraph")]
use std::collections::BTreeSet;

use super::DeviceRegistry;
use crate::{
    error::{QueryError, Result},
    plan::ResidentGraphKey,
};

/// Optional placement metadata for one query-service admission request.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryPlacementHint {
    /// Prefer devices that already retain this graph key in their local graph cache.
    pub resident_graph_key: Option<ResidentGraphKey>,
}

impl QueryPlacementHint {
    /// Creates an empty placement hint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a graph-cache affinity hint for a resident graph key.
    pub fn for_resident_graph_key(resident_graph_key: ResidentGraphKey) -> Self {
        Self { resident_graph_key: Some(resident_graph_key) }
    }

    /// Borrow the resident graph key requested by this hint, when any.
    pub fn resident_graph_key(&self) -> Option<&ResidentGraphKey> {
        self.resident_graph_key.as_ref()
    }

    /// Whether this hint carries no placement metadata.
    pub fn is_empty(&self) -> bool {
        self.resident_graph_key.is_none()
    }
}

/// Placement input for one admitted query.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlacementRequest {
    /// Minimum device-memory bytes required to place the query.
    pub min_device_budget_bytes: u64,
    /// Preferred device-memory bytes for placement.
    pub want_device_budget_bytes: u64,
    /// Optional tenant identifier for affinity policies.
    pub tenant_id: Option<String>,
    /// Optional query-local placement hints.
    pub placement_hint: QueryPlacementHint,
}

impl PlacementRequest {
    /// Creates a placement request for a fixed query budget.
    pub fn new(device_budget_bytes: u64) -> Self {
        Self {
            min_device_budget_bytes: device_budget_bytes,
            want_device_budget_bytes: device_budget_bytes,
            tenant_id: None,
            placement_hint: QueryPlacementHint::default(),
        }
    }

    /// Creates a placement request with separate minimum and preferred budgets.
    pub fn with_grant_range(min_device_budget_bytes: u64, want_device_budget_bytes: u64) -> Self {
        Self {
            min_device_budget_bytes,
            want_device_budget_bytes,
            tenant_id: None,
            placement_hint: QueryPlacementHint::default(),
        }
    }

    /// Adds tenant identity for policies that need tenant affinity.
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Adds optional placement metadata for policy composition.
    pub fn with_placement_hint(mut self, placement_hint: QueryPlacementHint) -> Self {
        self.placement_hint = placement_hint;
        self
    }
}

/// Policy that returns ordered candidate devices, most preferred first.
pub trait PlacementPolicy: Send + Sync {
    /// Ordered placement candidates for `request`.
    fn candidates(&self, registry: &DeviceRegistry, request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>>;
}

/// Configured placement policy for query-service admission.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryPlacementPolicy {
    /// Prefer the device with the most ungranted memory, then fewer active queries.
    #[default]
    LeastLoaded,
    /// Prefer the fullest device that can satisfy the query minimum.
    BestFit,
    /// Place every query on one configured CUDA ordinal.
    Pinned { device_ordinal: usize },
}

impl QueryPlacementPolicy {
    /// Stable snake-case label for diagnostics and config surfaces.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LeastLoaded => "least_loaded",
            Self::BestFit => "best_fit",
            Self::Pinned { .. } => "pinned",
        }
    }
}

impl PlacementPolicy for QueryPlacementPolicy {
    fn candidates(&self, registry: &DeviceRegistry, request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>> {
        match *self {
            Self::LeastLoaded => LeastLoadedPlacement.candidates(registry, request),
            Self::BestFit => BestFitPlacement.candidates(registry, request),
            Self::Pinned { device_ordinal } => {
                let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.query_service.placement",
                        "failed to resolve pinned query-service CUDA device",
                        error,
                    )
                    .with_fact("device_ordinal", device_ordinal)
                })?;
                PinnedDevicePlacement { device }.candidates(registry, request)
            },
        }
    }
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

/// Places queries on the fullest device that can satisfy the minimum grant.
#[derive(Debug, Default, Clone, Copy)]
pub struct BestFitPlacement;

impl PlacementPolicy for BestFitPlacement {
    fn candidates(&self, registry: &DeviceRegistry, request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>> {
        let mut loads = Vec::with_capacity(registry.len());
        for context in registry.contexts() {
            loads.push(DeviceLoad {
                ordinal: context.device().ordinal(),
                available_bytes: context.available_bytes()?,
                active_queries: context.active_queries(),
            });
        }
        Ok(rank_best_fit_device_loads(loads, request.min_device_budget_bytes)
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

fn rank_best_fit_device_loads(mut loads: Vec<DeviceLoad>, min_device_budget_bytes: u64) -> Vec<usize> {
    loads.sort_by(|left, right| {
        let left_fits = left.available_bytes >= min_device_budget_bytes;
        let right_fits = right.available_bytes >= min_device_budget_bytes;
        right_fits
            .cmp(&left_fits)
            .then_with(|| match (left_fits, right_fits) {
                (true, true) => left.available_bytes.cmp(&right.available_bytes),
                _ => right.available_bytes.cmp(&left.available_bytes),
            })
            .then_with(|| left.active_queries.cmp(&right.active_queries))
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });
    loads.into_iter().map(|load| load.ordinal).collect()
}

#[cfg(feature = "cugraph")]
pub(crate) fn rank_cache_affine_candidates(
    candidates: Vec<cudf::DeviceId>,
    cache_hit_ordinals: &BTreeSet<usize>,
) -> Vec<cudf::DeviceId> {
    rank_cache_affine_by(candidates, cache_hit_ordinals, |device| device.ordinal())
}

#[cfg(feature = "cugraph")]
fn rank_cache_affine_by<T>(
    candidates: Vec<T>,
    cache_hit_ordinals: &BTreeSet<usize>,
    ordinal: impl Fn(&T) -> usize,
) -> Vec<T> {
    if cache_hit_ordinals.is_empty() || candidates.len() <= 1 {
        return candidates;
    }
    let mut hits = Vec::new();
    let mut misses = Vec::new();
    for candidate in candidates {
        if cache_hit_ordinals.contains(&ordinal(&candidate)) {
            hits.push(candidate);
        } else {
            misses.push(candidate);
        }
    }
    hits.extend(misses);
    hits
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

    #[test]
    fn best_fit_ranking_prefers_fullest_device_that_can_fit_minimum() {
        let ranked = rank_best_fit_device_loads(
            vec![
                DeviceLoad { ordinal: 0, available_bytes: 80, active_queries: 3 },
                DeviceLoad { ordinal: 1, available_bytes: 30, active_queries: 0 },
                DeviceLoad { ordinal: 2, available_bytes: 60, active_queries: 1 },
                DeviceLoad { ordinal: 3, available_bytes: 60, active_queries: 0 },
            ],
            50,
        );

        assert_eq!(ranked, vec![3, 2, 0, 1]);
    }

    #[test]
    fn best_fit_ranking_keeps_non_fitting_devices_last_by_available_bytes() {
        let ranked = rank_best_fit_device_loads(
            vec![
                DeviceLoad { ordinal: 0, available_bytes: 30, active_queries: 0 },
                DeviceLoad { ordinal: 1, available_bytes: 20, active_queries: 0 },
                DeviceLoad { ordinal: 2, available_bytes: 40, active_queries: 0 },
            ],
            50,
        );

        assert_eq!(ranked, vec![2, 0, 1]);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_affine_ranking_stably_partitions_hit_devices() {
        let candidates = vec![0, 1, 2, 3];
        let hit_ordinals = BTreeSet::from([2, 0]);

        let ranked = rank_cache_affine_by(candidates, &hit_ordinals, |ordinal| *ordinal);

        assert_eq!(ranked, vec![0, 2, 1, 3]);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_affine_ranking_preserves_identity_without_hits() {
        let candidates = vec![1, 0];

        let ranked = rank_cache_affine_by(candidates, &BTreeSet::new(), |ordinal| *ordinal);

        assert_eq!(ranked, vec![1, 0]);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn cache_affine_ranking_leaves_single_candidate_unchanged() {
        let candidates = vec![1];

        let ranked = rank_cache_affine_by(candidates, &BTreeSet::from([1]), |ordinal| *ordinal);

        assert_eq!(ranked, vec![1]);
    }

    #[test]
    fn query_placement_policy_names_are_stable_snake_case() {
        assert_eq!(QueryPlacementPolicy::LeastLoaded.as_str(), "least_loaded");
        assert_eq!(QueryPlacementPolicy::BestFit.as_str(), "best_fit");
        assert_eq!(QueryPlacementPolicy::Pinned { device_ordinal: 2 }.as_str(), "pinned");
    }
}
