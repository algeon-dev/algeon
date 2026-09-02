//! Immutable per-device resource-controller profile.

use crate::{
    capability::{ContractStatus, ReservationRequirement},
    error::{QueryError, QueryErrorCode, Result},
    runtime::ReservationKind,
};

const OPERATION: &str = "nexus_query_engine.gpu.resource_profile";

/// Stable identity of one device-execution capability required by a native
/// attempt.
///
/// Capability requirements retain the reservation kind because the same
/// lower-level contract label can be meaningful at more than one execution
/// boundary.  The values are frozen into a device profile at installation;
/// placement never probes or mutates them at submission time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceCapability {
    reservation_kind: ReservationKind,
    required_contract: &'static str,
}

impl DeviceCapability {
    /// Creates one immutable device capability identity.
    pub const fn new(reservation_kind: ReservationKind, required_contract: &'static str) -> Self {
        Self { reservation_kind, required_contract }
    }

    fn from_requirement(requirement: &ReservationRequirement) -> Self {
        Self::new(requirement.kind, requirement.required_contract)
    }
}

/// Immutable capability metadata for one installed CUDA device.
///
/// Existing supported hardware uses [`Self::all_modeled`]: every capability
/// that the native planner has already modeled is available on that device.
/// Installers may instead freeze an exact list for heterogeneous deployments.
/// A non-modeled requirement is never accepted merely because it appears in
/// an exact list; the planner must establish the contract first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    support: DeviceCapabilitySupport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeviceCapabilitySupport {
    AllModeled,
    Exact(Vec<DeviceCapability>),
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self::all_modeled()
    }
}

impl DeviceCapabilities {
    /// Supports every capability that has a modeled native contract.
    pub const fn all_modeled() -> Self {
        Self { support: DeviceCapabilitySupport::AllModeled }
    }

    /// Supports only the supplied modeled capability identities.
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

    #[doc(hidden)]
    pub fn supports(&self, requirement: &ReservationRequirement) -> bool {
        if requirement.status != ContractStatus::Modeled {
            return false;
        }
        match &self.support {
            DeviceCapabilitySupport::AllModeled => true,
            DeviceCapabilitySupport::Exact(capabilities) => {
                capabilities.contains(&DeviceCapability::from_requirement(requirement))
            },
        }
    }

    #[doc(hidden)]
    pub fn supports_all(&self, requirements: &[ReservationRequirement]) -> bool {
        requirements.iter().all(|requirement| self.supports(requirement))
    }

    /// Whether admission must receive an explicit planner-produced requirement.
    ///
    /// An absent requirement is safe only when every modeled capability is
    /// accepted. Exact profiles must not interpret an empty implicit set as a
    /// proof that the attempt can execute on the device.
    #[doc(hidden)]
    pub const fn requires_explicit_attempt_requirement(&self) -> bool {
        matches!(self.support, DeviceCapabilitySupport::Exact(_))
    }
}

/// Frozen capacity and execution-slot bounds for one CUDA device.
///
/// `cache_cap_bytes` reserves a bounded per-device cache share alongside the
/// active-attempt floors within managed capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceResourceProfile {
    pub device_ordinal: usize,
    /// Capability metadata frozen with this device's resource envelope.
    pub device_capabilities: DeviceCapabilities,
    pub configured_limit_bytes: u64,
    pub headroom_bytes: u64,
    pub backend_reserve_bytes: u64,
    pub usable_capacity_bytes: u64,
    pub managed_capacity_bytes: u64,
    pub attempt_floor_bytes: u64,
    pub max_active_attempts: usize,
    pub root_lane_count: usize,
    pub root_stream_count: usize,
    pub source_work_per_attempt: usize,
    pub aux_source_work_capacity: usize,
    pub stream_pool_size: usize,
    pub worker_job_capacity: usize,
    pub cache_cap_bytes: u64,
}

impl DeviceResourceProfile {
    /// Resolves all resource-controller arithmetic for one device.
    pub fn resolve(
        device_ordinal: usize,
        configured_limit_bytes: u64,
        headroom_bytes: u64,
        backend_reserve_bytes: u64,
        attempt_floor_bytes: u64,
        max_active_attempts: usize,
        source_work_per_attempt: usize,
        stream_pool_size: usize,
        worker_job_capacity: usize,
    ) -> Result<Self> {
        Self::resolve_with_capabilities(
            device_ordinal,
            configured_limit_bytes,
            headroom_bytes,
            backend_reserve_bytes,
            attempt_floor_bytes,
            max_active_attempts,
            source_work_per_attempt,
            stream_pool_size,
            worker_job_capacity,
            DeviceCapabilities::default(),
        )
    }

    /// Resolves one profile with explicit immutable device capability metadata.
    pub fn resolve_with_capabilities(
        device_ordinal: usize,
        configured_limit_bytes: u64,
        headroom_bytes: u64,
        backend_reserve_bytes: u64,
        attempt_floor_bytes: u64,
        max_active_attempts: usize,
        source_work_per_attempt: usize,
        stream_pool_size: usize,
        worker_job_capacity: usize,
        device_capabilities: DeviceCapabilities,
    ) -> Result<Self> {
        Self::resolve_with_capabilities_and_cache_cap(
            device_ordinal,
            configured_limit_bytes,
            headroom_bytes,
            backend_reserve_bytes,
            attempt_floor_bytes,
            max_active_attempts,
            source_work_per_attempt,
            stream_pool_size,
            worker_job_capacity,
            device_capabilities,
            0,
        )
    }

    pub fn resolve_with_capabilities_and_cache_cap(
        device_ordinal: usize,
        configured_limit_bytes: u64,
        headroom_bytes: u64,
        backend_reserve_bytes: u64,
        attempt_floor_bytes: u64,
        max_active_attempts: usize,
        source_work_per_attempt: usize,
        stream_pool_size: usize,
        worker_job_capacity: usize,
        device_capabilities: DeviceCapabilities,
        cache_cap_bytes: u64,
    ) -> Result<Self> {
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
        if max_active_attempts == 0 {
            return Err(
                invalid(device_ordinal, "max_active_attempts", "must be greater than zero").with_fact("reason", "zero")
            );
        }
        if attempt_floor_bytes == 0 {
            return Err(
                invalid(device_ordinal, "attempt_floor_bytes", "must be greater than zero").with_fact("reason", "zero")
            );
        }
        let source_work_per_attempt = source_work_per_attempt.max(1);
        let active_attempts_u64 = u64::try_from(max_active_attempts).map_err(|_error| {
            invalid(device_ordinal, "max_active_attempts", "does not fit capacity arithmetic")
                .with_fact("reason", "out_of_range")
        })?;
        let required_floor = attempt_floor_bytes.checked_mul(active_attempts_u64).ok_or_else(|| {
            invalid(device_ordinal, "attempt_floor_bytes", "attempt floor multiplication overflowed")
                .with_fact("reason", "attempt_floor_overflow")
        })?;
        if required_floor > managed_capacity_bytes {
            return Err(invalid(device_ordinal, "attempt_floor_bytes", "attempt floors exceed managed capacity")
                .with_fact("reason", "attempt_floor_exceeds_managed_capacity")
                .with_fact("attempt_floor_total_bytes", required_floor)
                .with_fact("managed_capacity_bytes", managed_capacity_bytes));
        }
        let cache_attempt_floor = cache_cap_bytes.checked_add(required_floor).ok_or_else(|| {
            invalid(device_ordinal, "cache_cap_bytes", "cache cap and attempt floors overflow capacity arithmetic")
                .with_fact("reason", "cache_cap_overflow")
        })?;
        if cache_attempt_floor > managed_capacity_bytes {
            return Err(invalid(
                device_ordinal,
                "cache_cap_bytes",
                "cache cap plus attempt floors exceed managed capacity",
            )
            .with_fact("reason", "cache_cap_exceeds_managed_capacity")
            .with_fact("cache_cap_bytes", cache_cap_bytes)
            .with_fact("attempt_floor_total_bytes", required_floor)
            .with_fact("managed_capacity_bytes", managed_capacity_bytes));
        }
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
            attempt_floor_bytes,
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
    use super::*;
    use crate::capability::ContractQuality;

    #[test]
    fn resolves_heterogeneous_profiles_and_slot_math() {
        let small = DeviceResourceProfile::resolve(0, 1_000, 100, 100, 100, 2, 3, 8, 8).unwrap();
        let large = DeviceResourceProfile::resolve(1, 2_000, 200, 100, 100, 3, 1, 6, 6).unwrap();
        assert_eq!((small.usable_capacity_bytes, small.managed_capacity_bytes), (900, 800));
        assert_eq!((small.root_lane_count, small.root_stream_count, small.aux_source_work_capacity), (2, 2, 6));
        assert_eq!((large.usable_capacity_bytes, large.managed_capacity_bytes), (1_800, 1_700));
    }

    #[test]
    fn rejects_invalid_capacity_and_floor_arithmetic() {
        for (limit, headroom, reserve, reason) in [
            (99, 100, 0, "configured_limit_below_headroom"),
            (100, 0, 101, "backend_reserve_exceeds_usable_capacity"),
            (100, 0, 100, "zero_managed_capacity"),
        ] {
            let error = DeviceResourceProfile::resolve(7, limit, headroom, reserve, 0, 1, 1, 2, 2).unwrap_err();
            assert_eq!(error.fact_value("device_ordinal"), Some("7"));
            assert_eq!(error.fact_value("reason"), Some(reason));
        }
        let error = DeviceResourceProfile::resolve(7, u64::MAX, 0, 0, u64::MAX, 2, 1, 4, 4).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("attempt_floor_overflow"));
    }

    #[test]
    fn validates_cache_cap_against_managed_capacity_and_floor() {
        let profile = DeviceResourceProfile::resolve_with_capabilities_and_cache_cap(
            0,
            1_000,
            0,
            0,
            100,
            2,
            1,
            4,
            4,
            DeviceCapabilities::default(),
            800,
        )
        .unwrap();
        assert_eq!(profile.cache_cap_bytes, 800);
        let error = DeviceResourceProfile::resolve_with_capabilities_and_cache_cap(
            0,
            1_000,
            0,
            0,
            100,
            2,
            1,
            4,
            4,
            DeviceCapabilities::default(),
            801,
        )
        .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("cache_cap_exceeds_managed_capacity"));
    }

    #[test]
    fn rejects_zero_attempt_floor_for_an_otherwise_valid_profile() {
        let error = DeviceResourceProfile::resolve(7, 100, 0, 0, 0, 1, 1, 2, 2).unwrap_err();
        assert_eq!(error.fact_value("field"), Some("attempt_floor_bytes"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn rejects_floor_and_slot_capacity_below_requirements() {
        let floor = DeviceResourceProfile::resolve(3, 100, 0, 0, 51, 2, 1, 4, 4).unwrap_err();
        assert_eq!(floor.fact_value("reason"), Some("attempt_floor_exceeds_managed_capacity"));
        let slots = DeviceResourceProfile::resolve(3, 100, 0, 0, 10, 2, 3, 7, 7).unwrap_err();
        assert_eq!(slots.fact_value("reason"), Some("below_required_slot_capacity"));
    }

    #[test]
    fn frozen_capability_metadata_requires_modeled_contracts_and_exact_matches() {
        let modeled = ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "test_operator",
            "test_operator_contract",
            ContractQuality::CapabilityProof,
        );
        let blocked = ReservationRequirement::blocked_by_lower_level(
            ReservationKind::OperatorOutput,
            "test_operator",
            "test_operator_contract",
            ContractQuality::CapabilityProof,
        );
        let restricted = DeviceCapabilities::only([DeviceCapability::new(
            ReservationKind::OperatorOutput,
            "test_operator_contract",
        )]);

        assert!(restricted.supports(&modeled));
        assert!(!restricted.supports(&blocked));
        assert!(!DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()).supports(&modeled));
    }
}
