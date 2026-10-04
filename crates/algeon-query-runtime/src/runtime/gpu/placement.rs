//! Placement policies for independent query scheduling across devices.
//!
//! Placement consumes one atomic ledger snapshot per device. It never reads
//! NVML or live CUDA memory samples: configured capacity, immutable grants,
//! reservations, cache residency, and ledger health drive
//! admission. Native-health reconciliation belongs to the query-service
//! boundary before queue maintenance or direct placement begins; this module
//! only reads the resulting ledger state.

use query_planning::{
    admission::{AttemptRequirement, ExecutionCapability},
    error::Result,
};

#[cfg(test)]
use super::DeviceResourceEnvelope;
use super::{DeviceAdmissionHealth, DeviceLedger, DeviceResourceProfile};

/// Internal placement input derived from planner-owned capability evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct PlacementRequest {
    /// Explicit query cap. `None` lets each device resolve its configured
    /// default against the same resident snapshot used for admission.
    pub(crate) requested_query_cap_bytes: Option<u64>,
    /// Device-independent requirement resolved once per candidate ledger.
    pub(crate) attempt_requirement: Option<AttemptRequirement>,
    /// Device capabilities that must be present before this request can enter
    /// capacity ranking. These are a safety predicate, not a placement hint.
    pub(crate) required_capabilities: Vec<ExecutionCapability>,
}

impl PlacementRequest {
    /// Creates a placement request. Zero selects the configured default.
    pub(crate) fn new(query_cap_bytes: u64) -> Self {
        Self {
            requested_query_cap_bytes: (query_cap_bytes > 0).then_some(query_cap_bytes),
            attempt_requirement: None,
            required_capabilities: Vec::new(),
        }
    }

    /// Retains capability evidence for device executability filtering.
    pub(crate) fn with_attempt_requirement(mut self, attempt_requirement: AttemptRequirement) -> Self {
        self.attempt_requirement = Some(attempt_requirement);
        self
    }

    /// Adds immutable device capabilities required by the attempt.
    pub(crate) fn with_required_capabilities(
        mut self,
        required_capabilities: impl IntoIterator<Item = ExecutionCapability>,
    ) -> Self {
        self.required_capabilities = required_capabilities.into_iter().collect();
        self
    }

    pub(crate) fn with_requested_query_cap_bytes(mut self, requested_query_cap_bytes: Option<u64>) -> Self {
        self.requested_query_cap_bytes = requested_query_cap_bytes;
        self
    }
}

/// One candidate device's placement view at one ledger observation sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacementCandidate {
    pub(crate) ordinal: usize,
    /// Cap resolved against this candidate's resident snapshot.
    pub(crate) query_cap_bytes: u64,
    /// Capacity remaining after this admission.
    pub(crate) spare_capacity_bytes: u64,
    pub(crate) active_attempts: usize,
    pub(crate) live_grants_bytes: u64,
    /// Logical worker slots committed by active attempts in the atomic ledger
    /// snapshot.
    pub(crate) worker_depth: usize,
    /// Logical stream slots still available in that same snapshot.
    pub(crate) stream_slots_available: usize,
}

/// One atomic ledger observation used for both queue and direct placement.
///
/// `candidate` contains every ranking input from the same snapshot;
/// `can_admit_now` states whether that snapshot has enough capacity for the
/// request.  A later `try_admit` remains the authoritative reservation
/// transaction, so this helper never turns a ranking observation into a
/// second admission authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacementCandidateSnapshot {
    pub(crate) candidate: PlacementCandidate,
    pub(crate) can_admit_now: bool,
    /// Whether the immutable device profile can ever satisfy this candidate's
    /// resolved floor, independent of concurrent grants. This belongs beside
    /// the snapshot so callers do not reconstruct a different permanent-
    /// capacity classification after ranking.
    pub(crate) can_ever_fit_query_cap: bool,
}

impl PlacementCandidate {
    /// Spare bytes beyond this candidate's required floor after admission.
    fn spare_after(&self) -> u64 {
        self.spare_capacity_bytes
    }
}

/// Documented stable placement ranking:
///
/// 1. spare capacity after admission, descending (least-loaded);
/// 2. active attempts, ascending;
/// 3. live grants, ascending;
/// 4. worker depth, ascending;
/// 5. stream availability, descending;
/// 6. device ordinal, ascending (deterministic tie-break).
pub(crate) fn compare_placement_candidates(
    left: &PlacementCandidate,
    right: &PlacementCandidate,
) -> std::cmp::Ordering {
    right
        .spare_after()
        .cmp(&left.spare_after())
        .then_with(|| left.active_attempts.cmp(&right.active_attempts))
        .then_with(|| left.live_grants_bytes.cmp(&right.live_grants_bytes))
        .then_with(|| left.worker_depth.cmp(&right.worker_depth))
        .then_with(|| right.stream_slots_available.cmp(&left.stream_slots_available))
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

#[cfg(test)]
pub(crate) fn rank_placement_candidates(mut candidates: Vec<PlacementCandidate>, query_cap_bytes: u64) -> Vec<usize> {
    for candidate in &mut candidates {
        candidate.query_cap_bytes = query_cap_bytes;
    }
    candidates.sort_by(compare_placement_candidates);
    candidates.into_iter().map(|candidate| candidate.ordinal).collect()
}

/// Captures one ledger's ranking evidence from one atomic snapshot.
///
/// Returns `None` when the ledger cannot serve the request at all: its
/// profile lacks a required capability or its admission is closed. A
/// saturated open ledger stays a candidate so placement can distinguish a
/// transiently busy device from one that can never fit the required floor.
pub(crate) fn placement_candidate_snapshot(
    ledger: &DeviceLedger,
    request: &PlacementRequest,
) -> Result<Option<PlacementCandidateSnapshot>> {
    if !profile_supports_placement_request(ledger.profile(), &request.required_capabilities) {
        return Ok(None);
    }
    let snapshot = ledger.snapshot()?;
    if !matches!(snapshot.health, DeviceAdmissionHealth::Open) {
        return Ok(None);
    }
    let query_cap_bytes = request.requested_query_cap_bytes.unwrap_or_else(|| {
        snapshot
            .default_query_limit_bytes
            .min(snapshot.query_share_bytes.saturating_sub(snapshot.frozen_resident_bytes))
    });
    let can_ever_fit_query_cap = snapshot.can_ever_fit_query_cap(request.requested_query_cap_bytes);
    let admitted_cap = snapshot.admissible_query_cap_bytes(request.requested_query_cap_bytes);
    let spare_capacity_bytes = admitted_cap.map_or(0, |cap| {
        snapshot
            .query_share_bytes
            .saturating_sub(snapshot.frozen_resident_bytes)
            .saturating_sub(snapshot.live_grants_bytes)
            .saturating_sub(cap)
    });
    Ok(Some(PlacementCandidateSnapshot {
        candidate: PlacementCandidate {
            ordinal: ledger.device_ordinal(),
            query_cap_bytes,
            spare_capacity_bytes,
            active_attempts: snapshot.active_attempts,
            live_grants_bytes: snapshot.live_grants_bytes,
            worker_depth: snapshot.worker_depth,
            stream_slots_available: snapshot.stream_slots_available(),
        },
        can_admit_now: admitted_cap.is_some(),
        can_ever_fit_query_cap,
    }))
}

/// The stable placement order: candidates that can admit now come first,
/// then the documented ranking of [`compare_placement_candidates`].
pub(crate) fn compare_placement_candidate_snapshots(
    left: &PlacementCandidateSnapshot,
    right: &PlacementCandidateSnapshot,
) -> std::cmp::Ordering {
    right
        .can_admit_now
        .cmp(&left.can_admit_now)
        .then_with(|| compare_placement_candidates(&left.candidate, &right.candidate))
}

/// Applies the immutable capability gate before a candidate can be observed
/// by the ledger ranking. Keeping this predicate separate makes it explicit
/// that no capacity or load signal may resurrect an incapable device.
fn profile_supports_placement_request(
    profile: &DeviceResourceProfile,
    required_capabilities: &[ExecutionCapability],
) -> bool {
    profile.device_capabilities.supports_all(required_capabilities)
}

#[cfg(test)]
mod tests {
    use query_planning::admission::ExecutionCapability;

    use super::*;
    use crate::runtime::{DeviceCapabilities, DeviceCapability};

    fn candidate(
        ordinal: usize,
        max_new_grant: u64,
        active: usize,
        live: u64,
        depth: usize,
        streams: usize,
    ) -> PlacementCandidate {
        PlacementCandidate {
            ordinal,
            query_cap_bytes: 0,
            spare_capacity_bytes: max_new_grant,
            active_attempts: active,
            live_grants_bytes: live,
            worker_depth: depth,
            stream_slots_available: streams,
        }
    }

    #[test]
    fn placement_ranking_prefers_spare_then_active_then_grants_then_depth_then_streams_then_ordinal() {
        let ranked = rank_placement_candidates(
            vec![
                candidate(2, 100, 0, 0, 0, 10),
                candidate(1, 200, 2, 10, 0, 10),
                candidate(0, 200, 1, 50, 0, 10),
                candidate(3, 200, 1, 50, 0, 20),
                candidate(4, 200, 1, 40, 1, 30),
            ],
            50,
        );

        assert_eq!(ranked, vec![4, 3, 0, 1, 2]);
    }

    #[test]
    fn placement_ranking_is_stable_for_heterogeneous_envelopes() {
        let ranked =
            rank_placement_candidates(vec![candidate(0, 1_000, 0, 0, 0, 10), candidate(1, 500, 0, 0, 0, 10)], 400);

        assert_eq!(ranked, vec![0, 1], "the larger envelope offers more spare capacity");
    }

    #[test]
    fn capability_gate_excludes_incapable_profiles_before_placement_ranking() {
        let requirement = ExecutionCapability::new("test_operator");
        let incapable = /*OK*/DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 0,
            configured_limit_bytes: 1_000,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 100,
            max_active_attempts: 1,
            source_work_per_attempt: 1,
            stream_pool_size: 2,
            worker_job_capacity: 2,
            device_capabilities: DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()),
            cache_cap_bytes: 0,
        })
        .unwrap();
        let capable = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
            device_ordinal: 1,
            configured_limit_bytes: 1_000,
            headroom_bytes: 0,
            backend_reserve_bytes: 0,
            default_query_limit_bytes: 100,
            max_active_attempts: 1,
            source_work_per_attempt: 1,
            stream_pool_size: 2,
            worker_job_capacity: 2,
            device_capabilities: DeviceCapabilities::only([DeviceCapability::new("test_operator")]),
            cache_cap_bytes: 0,
        })
        .unwrap();

        assert!(!profile_supports_placement_request(&incapable, std::slice::from_ref(&requirement)));
        assert!(profile_supports_placement_request(&capable, &[requirement]));
    }
}
