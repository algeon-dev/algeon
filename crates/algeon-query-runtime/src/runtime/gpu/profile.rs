//! Immutable per-device resource-controller profile.

use query_planning::{
    admission::ExecutionCapability,
    error::{QueryError, QueryErrorCode, Result},
};

const OPERATION: &str = "query_engine.gpu.resource_profile";

/// Stable identity of one device-execution capability required by a native
/// attempt.
///
/// Values are frozen into a device profile at installation; placement never
/// probes or mutates them at submission time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceCapability {
    capability: ExecutionCapability,
}

impl DeviceCapability {
    /// Creates one immutable device capability identity.
    pub const fn new(name: &'static str) -> Self {
        Self { capability: ExecutionCapability::new(name) }
    }

    const fn from_requirement(requirement: ExecutionCapability) -> Self {
        Self { capability: requirement }
    }
}

/// Immutable capability metadata for one installed CUDA device.
///
/// Existing supported hardware declares every execution capability available on
/// that device.
/// Installers may instead freeze an exact list for heterogeneous deployments.
/// A non-modeled requirement is never accepted merely because it appears in
/// an exact list; the planner must establish the contract first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    support: DeviceCapabilitySupport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeviceCapabilitySupport {
    All,
    Exact(Vec<DeviceCapability>),
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self::all()
    }
}

impl DeviceCapabilities {
    /// Supports every declared execution capability.
    pub(crate) const fn all() -> Self {
        Self { support: DeviceCapabilitySupport::All }
    }

    /// Supports only the supplied capability identities.
    ///
    /// This is the per-device construction input for heterogeneous
    /// deployments.  Duplicate entries are harmless and do not change
    /// placement semantics.
    pub fn only(capabilities: impl IntoIterator<Item = DeviceCapability>) -> Self {
        let mut exact = Vec::new();
        for capability in capabilities {
            if !exact.contains(&capability) {
                exact.push(capability);
            }
        }
        Self { support: DeviceCapabilitySupport::Exact(exact) }
    }

    pub(crate) fn supports(&self, requirement: &ExecutionCapability) -> bool {
        match &self.support {
            DeviceCapabilitySupport::All => true,
            DeviceCapabilitySupport::Exact(capabilities) => {
                capabilities.contains(&DeviceCapability::from_requirement(*requirement))
            },
        }
    }

    pub(crate) fn supports_all(&self, requirements: &[ExecutionCapability]) -> bool {
        requirements.iter().all(|requirement| self.supports(requirement))
    }

    /// Whether admission must receive an explicit planner-produced requirement.
    ///
    /// An absent requirement is safe only when every execution capability is
    /// accepted. Exact profiles must not interpret an empty implicit set as a
    /// evidence that the attempt can execute on the device.
    pub(crate) const fn requires_explicit_attempt_requirement(&self) -> bool {
        matches!(self.support, DeviceCapabilitySupport::Exact(_))
    }
}

/// Frozen capacity and execution-slot bounds for one CUDA device.
///
/// `cache_cap_bytes` (`K`) reserves a bounded per-device cache share carved
/// out of the managed capacity (`M`). Query grants and frozen residents share
/// only the remaining query share (`Q = M - K`); query admission never
/// promises the same bytes that back the cache share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceResourceProfile {
    pub device_ordinal: usize,
    /// Capability metadata frozen with this device's resource envelope.
    pub device_capabilities: DeviceCapabilities,
    pub(crate) configured_limit_bytes: u64,
    pub(crate) headroom_bytes: u64,
    pub(crate) backend_reserve_bytes: u64,
    pub(crate) usable_capacity_bytes: u64,
    pub managed_capacity_bytes: u64,
    pub default_query_limit_bytes: u64,
    pub max_active_attempts: usize,
    pub root_lane_count: usize,
    pub root_stream_count: usize,
    pub source_work_per_attempt: usize,
    pub(crate) aux_source_work_capacity: usize,
    pub stream_pool_size: usize,
    pub worker_job_capacity: usize,
    pub cache_cap_bytes: u64,
}

/// Immutable input contract for one device's resolved resource envelope.
///
/// These fields jointly define the admission envelope one device freezes at
/// installation: capacity splits, the default query limit, sloting, and cache
/// share. `resolve` consumes the envelope; the derived capacities, lane and
/// stream counts, and aux-source-work arithmetic stay computed outputs on
/// [`DeviceResourceProfile`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceResourceEnvelope {
    /// Device ordinal this envelope resolves for.
    pub(crate) device_ordinal: usize,
    /// Capability metadata frozen with this device's resource envelope.
    pub(crate) device_capabilities: DeviceCapabilities,
    /// Total configured allocation limit before headroom.
    pub(crate) configured_limit_bytes: u64,
    /// Headroom subtracted from the configured limit before backend reserve.
    pub(crate) headroom_bytes: u64,
    /// Backend-reserved bytes unavailable to managed attempts.
    pub(crate) backend_reserve_bytes: u64,
    /// Default per-query limit guaranteed inside managed capacity.
    pub(crate) default_query_limit_bytes: u64,
    /// Maximum simultaneously admitted attempts.
    pub(crate) max_active_attempts: usize,
    /// Auxiliary slots granted per attempt.
    ///
    /// Each slot backs one auxiliary permit holder of the attempt: an
    /// execution lane, a source worker, or an auxiliary fragment context.
    /// The native route serves one disjoint fork subset per live holder, so
    /// this count sizes the attempt's fork pool, not its operator width.
    /// Execution width is the minimum of this envelope, the free permits,
    /// the device execution-job capacity, and the ready work.
    pub(crate) source_work_per_attempt: usize,
    /// Installed stream-pool size; must cover root and auxiliary work.
    pub(crate) stream_pool_size: usize,
    /// Installed worker-job capacity; must cover root and auxiliary work.
    pub(crate) worker_job_capacity: usize,
    /// Bounded per-device cache share reserved alongside the default query limit.
    pub(crate) cache_cap_bytes: u64,
}

impl DeviceResourceProfile {
    /// Query share (`Q = M - K`) promisable to grants plus frozen residents.
    ///
    /// Cache charges (`C <= K`) never participate in this share, so query
    /// admission and cache eviction cannot promise the same bytes.
    pub fn query_share_bytes(&self) -> u64 {
        self.managed_capacity_bytes.saturating_sub(self.cache_cap_bytes)
    }

    /// Derives the default per-query grant from the service parallelism
    /// target: `floor((M - K) / N)` with checked arithmetic.
    ///
    /// This is a capacity-allocation policy, not a per-query estimate: with
    /// no frozen residents the profile admits `N` default grants. An
    /// explicitly configured default keeps its value after validation.
    pub(crate) fn derive_default_query_limit_bytes(
        device_ordinal: usize,
        managed_capacity_bytes: u64,
        cache_cap_bytes: u64,
        max_active_attempts: usize,
    ) -> Result<u64> {
        if max_active_attempts == 0 {
            return Err(
                invalid(device_ordinal, "max_active_attempts", "must be greater than zero").with_fact("reason", "zero")
            );
        }
        let query_share_bytes =
            managed_capacity_bytes.checked_sub(cache_cap_bytes).filter(|share| *share > 0).ok_or_else(|| {
                invalid(device_ordinal, "cache_cap_bytes", "cache cap must leave a non-zero query share")
                    .with_fact("reason", "cache_cap_exceeds_managed_capacity")
                    .with_fact("cache_cap_bytes", cache_cap_bytes)
                    .with_fact("managed_capacity_bytes", managed_capacity_bytes)
            })?;
        let divisor = u64::try_from(max_active_attempts).map_err(|_| {
            invalid(device_ordinal, "max_active_attempts", "parallelism target exceeds the capacity integer range")
                .with_fact("reason", "max_active_attempts_overflow")
        })?;
        let derived = query_share_bytes.checked_div(divisor).filter(|cap| *cap > 0).ok_or_else(|| {
            invalid(device_ordinal, "default_query_limit_bytes", "derived default query limit is zero")
                .with_fact("reason", "derived_default_query_limit_is_zero")
                .with_fact("query_share_bytes", query_share_bytes)
                .with_fact("max_active_attempts", max_active_attempts)
        })?;
        Ok(derived)
    }

    /// Resolves all resource-controller arithmetic for one device from its
    /// immutable resource envelope.
    pub(crate) fn resolve(envelope: DeviceResourceEnvelope) -> Result<Self> {
        let DeviceResourceEnvelope {
            device_ordinal,
            device_capabilities,
            configured_limit_bytes,
            headroom_bytes,
            backend_reserve_bytes,
            default_query_limit_bytes,
            max_active_attempts,
            source_work_per_attempt,
            stream_pool_size,
            worker_job_capacity,
            cache_cap_bytes,
        } = envelope;
        let usable_capacity_bytes = configured_limit_bytes.checked_sub(headroom_bytes).ok_or_else(|| {
            invalid(device_ordinal, "headroom_bytes", "configured limit is below reserved headroom")
                .with_fact("reason", "configured_limit_below_headroom")
                .with_fact("configured_limit_bytes", configured_limit_bytes)
                .with_fact("headroom_bytes", headroom_bytes)
        })?;
        let managed_capacity_bytes = usable_capacity_bytes.checked_sub(backend_reserve_bytes).ok_or_else(|| {
            invalid(device_ordinal, "backend_reserve_bytes", "backend reserve exceeds usable capacity")
                .with_fact("reason", "backend_reserve_exceeds_usable_capacity")
                .with_fact("usable_capacity_bytes", usable_capacity_bytes)
                .with_fact("backend_reserve_bytes", backend_reserve_bytes)
        })?;
        if managed_capacity_bytes == 0 {
            return Err(invalid(
                device_ordinal,
                "managed_capacity_bytes",
                "managed capacity must be greater than zero",
            )
            .with_fact("reason", "zero_managed_capacity"));
        }
        if cache_cap_bytes >= managed_capacity_bytes {
            return Err(invalid(
                device_ordinal,
                "cache_cap_bytes",
                "cache cap must leave a non-zero query share of managed capacity",
            )
            .with_fact("reason", "cache_cap_exceeds_managed_capacity")
            .with_fact("cache_cap_bytes", cache_cap_bytes)
            .with_fact("managed_capacity_bytes", managed_capacity_bytes));
        }
        if max_active_attempts == 0 {
            return Err(
                invalid(device_ordinal, "max_active_attempts", "must be greater than zero").with_fact("reason", "zero")
            );
        }
        if default_query_limit_bytes == 0 {
            return Err(invalid(device_ordinal, "default_query_limit_bytes", "must be greater than zero")
                .with_fact("reason", "zero"));
        }
        let query_share_bytes = managed_capacity_bytes.checked_sub(cache_cap_bytes).ok_or_else(|| {
            invalid(device_ordinal, "cache_cap_bytes", "cache cap and query share overflow capacity arithmetic")
                .with_fact("reason", "cache_cap_overflow")
        })?;
        if default_query_limit_bytes > query_share_bytes {
            return Err(invalid(
                device_ordinal,
                "default_query_limit_bytes",
                "default query limit exceeds the query share of managed capacity",
            )
            .with_fact("reason", "default_query_limit_exceeds_managed_capacity")
            .with_fact("default_query_limit_bytes", default_query_limit_bytes)
            .with_fact("query_share_bytes", query_share_bytes)
            .with_fact("managed_capacity_bytes", managed_capacity_bytes));
        }
        let source_work_per_attempt = source_work_per_attempt.max(1);
        let aux_source_work_capacity = max_active_attempts.checked_mul(source_work_per_attempt).ok_or_else(|| {
            invalid(device_ordinal, "source_work_per_attempt", "auxiliary source-work capacity overflowed")
                .with_fact("reason", "aux_source_work_overflow")
        })?;
        let required_slots = max_active_attempts.checked_add(aux_source_work_capacity).ok_or_else(|| {
            invalid(device_ordinal, "stream_pool_size", "required stream capacity overflowed")
                .with_fact("reason", "required_slot_overflow")
        })?;
        for (field, actual) in [("stream_pool_size", stream_pool_size), ("worker_job_capacity", worker_job_capacity)] {
            if actual < required_slots {
                return Err(invalid(device_ordinal, field, "must cover root and auxiliary source work")
                    .with_fact("reason", "below_required_slot_capacity")
                    .with_fact("required_slot_capacity", required_slots)
                    .with_fact("actual_capacity", actual));
            }
        }
        Ok(Self {
            device_ordinal,
            device_capabilities,
            configured_limit_bytes,
            headroom_bytes,
            backend_reserve_bytes,
            usable_capacity_bytes,
            managed_capacity_bytes,
            default_query_limit_bytes,
            max_active_attempts,
            root_lane_count: max_active_attempts,
            root_stream_count: max_active_attempts,
            source_work_per_attempt,
            aux_source_work_capacity,
            stream_pool_size,
            worker_job_capacity,
            cache_cap_bytes,
        })
    }
}

fn invalid(device_ordinal: usize, field: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("field", field)
}

#[cfg(test)]
mod tests {
    use query_planning::admission::ExecutionCapability;

    use super::*;

    #[test]
    fn resolves_heterogeneous_profiles_and_slot_math() {
        let small = /*OK*/DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 0,
            configured_limit_bytes: 1_000,
            headroom_bytes: 100,
            backend_reserve_bytes: 100,
            default_query_limit_bytes: 100,
            max_active_attempts: 2,
            source_work_per_attempt: 3,
            stream_pool_size: 8,
            worker_job_capacity: 8,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 0,
        }).unwrap();
        let large = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 1,
            configured_limit_bytes: 2_000,
            headroom_bytes: 200,
            backend_reserve_bytes: 100,
            default_query_limit_bytes: 100,
            max_active_attempts: 3,
            source_work_per_attempt: 1,
            stream_pool_size: 6,
            worker_job_capacity: 6,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 0,
        })
        .unwrap();
        assert_eq!((small.usable_capacity_bytes, small.managed_capacity_bytes), (900, 800));
        assert_eq!((small.root_lane_count, small.root_stream_count, small.aux_source_work_capacity), (2, 2, 6));
        assert_eq!((large.usable_capacity_bytes, large.managed_capacity_bytes), (1_800, 1_700));
    }

    #[test]
    fn rejects_invalid_capacity_and_query_limit_arithmetic() {
        for (limit, headroom, reserve, reason) in [
            (99, 100, 0, "configured_limit_below_headroom"),
            (100, 0, 101, "backend_reserve_exceeds_usable_capacity"),
            (100, 0, 100, "zero_managed_capacity"),
        ] {
            let error = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
                device_ordinal: 7,
                configured_limit_bytes: limit,
                headroom_bytes: headroom,
                backend_reserve_bytes: reserve,
                default_query_limit_bytes: 0,
                max_active_attempts: 1,
                source_work_per_attempt: 1,
                stream_pool_size: 2,
                worker_job_capacity: 2,
                device_capabilities: DeviceCapabilities::default(),
                cache_cap_bytes: 0,
            })
            .unwrap_err();
            assert_eq!(error.fact_value("device_ordinal"), Some("7"));
            assert_eq!(error.fact_value("reason"), Some(reason));
        }
        let error = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 7,
            configured_limit_bytes: u64::MAX - 1,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: u64::MAX,
            max_active_attempts: 2,
            source_work_per_attempt: 1,
            stream_pool_size: 4,
            worker_job_capacity: 4,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 0,
        })
        .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("default_query_limit_exceeds_managed_capacity"));
    }

    #[test]
    fn validates_cache_cap_against_managed_capacity() {
        let profile = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 0,
            configured_limit_bytes: 1_000,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 100,
            max_active_attempts: 2,
            source_work_per_attempt: 1,
            stream_pool_size: 4,
            worker_job_capacity: 4,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 800,
        })
        .unwrap();
        assert_eq!(profile.cache_cap_bytes, 800);
        let error = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 0,
            configured_limit_bytes: 1_000,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 100,
            max_active_attempts: 2,
            source_work_per_attempt: 1,
            stream_pool_size: 4,
            worker_job_capacity: 4,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 1_001,
        })
        .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("cache_cap_exceeds_managed_capacity"));
    }

    #[test]
    fn rejects_zero_default_query_limit_for_an_otherwise_valid_profile() {
        let error = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 7,
            configured_limit_bytes: 100,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 0,
            max_active_attempts: 1,
            source_work_per_attempt: 1,
            stream_pool_size: 2,
            worker_job_capacity: 2,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 0,
        })
        .unwrap_err();
        assert_eq!(error.fact_value("field"), Some("default_query_limit_bytes"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn rejects_query_limit_and_slot_capacity_below_requirements() {
        let query_limit = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 3,
            configured_limit_bytes: 100,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 101,
            max_active_attempts: 2,
            source_work_per_attempt: 1,
            stream_pool_size: 4,
            worker_job_capacity: 4,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 0,
        })
        .unwrap_err();
        assert_eq!(query_limit.fact_value("reason"), Some("default_query_limit_exceeds_managed_capacity"));
        let slots = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 3,
            configured_limit_bytes: 100,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 10,
            max_active_attempts: 2,
            source_work_per_attempt: 3,
            stream_pool_size: 7,
            worker_job_capacity: 7,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes: 0,
        })
        .unwrap_err();
        assert_eq!(slots.fact_value("reason"), Some("below_required_slot_capacity"));
    }

    #[test]
    fn frozen_capability_metadata_requires_exact_matches() {
        let required = ExecutionCapability::new("test_operator");
        let other = ExecutionCapability::new("other_operator");
        let restricted = DeviceCapabilities::only([DeviceCapability::new("test_operator")]);

        assert!(restricted.supports(&required));
        assert!(!restricted.supports(&other));
        assert!(!DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()).supports(&required));
    }

    #[test]
    fn derives_default_query_limit_from_parallelism_target_and_cache_share() {
        // `floor((M - K) / N)`: the plan's `M = 24, K = 8, N = 2` fixture
        // yields 8, so two defaults exactly fill the query share.
        assert_eq!(DeviceResourceProfile::derive_default_query_limit_bytes(0, 24, 8, 2).unwrap(), 8);
        assert_eq!(DeviceResourceProfile::derive_default_query_limit_bytes(0, 10, 0, 3).unwrap(), 3);
        assert_eq!(DeviceResourceProfile::derive_default_query_limit_bytes(0, 1_000, 0, 1).unwrap(), 1_000);

        let zero_share = DeviceResourceProfile::derive_default_query_limit_bytes(0, 10, 9, 3).unwrap_err();
        assert_eq!(zero_share.fact_value("reason"), Some("derived_default_query_limit_is_zero"));

        let no_share = DeviceResourceProfile::derive_default_query_limit_bytes(0, 8, 8, 2).unwrap_err();
        assert_eq!(no_share.fact_value("reason"), Some("cache_cap_exceeds_managed_capacity"));

        let no_parallelism = DeviceResourceProfile::derive_default_query_limit_bytes(0, 24, 8, 0).unwrap_err();
        assert_eq!(no_parallelism.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn query_share_subtracts_the_cache_share() {
        let envelope = |cache_cap_bytes, default_query_limit_bytes| DeviceResourceEnvelope {
            device_ordinal: 0,
            configured_limit_bytes: 1_000,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes,
            max_active_attempts: 2,
            source_work_per_attempt: 1,
            stream_pool_size: 4,
            worker_job_capacity: 4,
            device_capabilities: DeviceCapabilities::default(),
            cache_cap_bytes,
        };
        let profile = DeviceResourceProfile::resolve(envelope(300, 700)).unwrap();
        assert_eq!(profile.query_share_bytes(), 700);

        let beyond_share = DeviceResourceProfile::resolve(envelope(300, 701)).unwrap_err();
        assert_eq!(beyond_share.fact_value("reason"), Some("default_query_limit_exceeds_managed_capacity"));

        let no_share = DeviceResourceProfile::resolve(envelope(1_000, 1)).unwrap_err();
        assert_eq!(no_share.fact_value("reason"), Some("cache_cap_exceeds_managed_capacity"));
    }
}
