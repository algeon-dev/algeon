//! Placement policies for independent query scheduling across devices.
//!
//! Placement consumes one atomic ledger snapshot per device. It never reads
//! NVML or live CUDA memory samples: configured capacity, immutable grants,
//! reservations, cache residency (zero in P2), and ledger health drive
//! admission. Native-health reconciliation belongs to the query-service
//! boundary before queue maintenance or direct placement begins; this module
//! only reads the resulting ledger state.

use super::{DeviceLedger, DeviceRegistry, DeviceResourceProfile};
use crate::{admission::AttemptRequirement, capability::ReservationRequirement, error::Result};

/// Internal placement input derived from planner-owned capability evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct PlacementRequest {
    /// Explicit required attempt floor proven before admission.
    ///
    /// This is retained for callers that already resolved a floor. New
    /// query-service placement supplies [`Self::attempt_requirement`] so a
    /// zero request floor resolves against each candidate profile.
    pub(crate) required_floor_bytes: u64,
    /// Device-independent requirement resolved once per candidate ledger.
    pub(crate) attempt_requirement: Option<AttemptRequirement>,
    /// Device capabilities that must be present before this request can enter
    /// capacity ranking. These are a safety predicate, not a placement hint.
    pub(crate) required_capabilities: Vec<ReservationRequirement>,
}

impl PlacementRequest {
    /// Creates a placement request for a required attempt floor.
    pub(crate) fn new(required_floor_bytes: u64) -> Self {
        Self { required_floor_bytes, attempt_requirement: None, required_capabilities: Vec::new() }
    }

    /// Resolves a zero configured floor using the candidate device profile.
    pub(crate) fn with_attempt_requirement(mut self, attempt_requirement: AttemptRequirement) -> Self {
        self.attempt_requirement = Some(attempt_requirement);
        self
    }

    /// Adds immutable device capabilities required by the attempt.
    pub(crate) fn with_required_capabilities(
        mut self,
        required_capabilities: impl IntoIterator<Item = ReservationRequirement>,
    ) -> Self {
        self.required_capabilities = required_capabilities.into_iter().collect();
        self
    }

    pub(crate) fn required_floor_bytes_for(&self, ledger: &DeviceLedger) -> Result<u64> {
        self.attempt_requirement
            .as_ref()
            .map_or(Ok(self.required_floor_bytes), |requirement| ledger.required_floor_bytes_for(requirement))
    }
}

/// One candidate device's placement view at one ledger observation sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacementCandidate {
    pub(crate) ordinal: usize,
    /// Floor resolved against this candidate's immutable profile.
    pub(crate) required_floor_bytes: u64,
    /// Headroom the normative equation offers a new grant on this device.
    pub(crate) max_new_grant_bytes: u64,
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
    pub(crate) can_ever_fit_required_floor: bool,
}

impl PlacementCandidate {
    /// Spare bytes beyond this candidate's required floor after admission.
    fn spare_after(&self) -> u64 {
        self.max_new_grant_bytes.saturating_sub(self.required_floor_bytes)
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

/// Selects the single best candidate with the documented placement order.
///
/// Both direct placement and queue admission call this helper so that a
/// queue waiter cannot silently diverge from normal device ranking.
pub(crate) fn preferred_placement_candidate(
    candidates: impl IntoIterator<Item = PlacementCandidate>,
) -> Option<PlacementCandidate> {
    candidates.into_iter().min_by(compare_placement_candidates)
}

#[cfg(test)]
pub(crate) fn rank_placement_candidates(
    mut candidates: Vec<PlacementCandidate>,
    required_floor_bytes: u64,
) -> Vec<usize> {
    for candidate in &mut candidates {
        candidate.required_floor_bytes = required_floor_bytes;
    }
    candidates.sort_by(compare_placement_candidates);
    candidates.into_iter().map(|candidate| candidate.ordinal).collect()
}

/// Policy that returns ordered candidate devices, most preferred first.
pub(crate) trait PlacementPolicy: Send + Sync {
    /// Ordered placement candidates for `request`.
    fn candidates(&self, registry: &DeviceRegistry, request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>>;
}

/// Captures candidate ranking evidence from one ledger snapshot per ordinal.
///
/// Queue selection passes a protected ordinal when its finite-overtake
/// reservation is eligible for replacement; direct placement passes `None`.
/// The reservation is still consumed only by the ledger's admission
/// transaction, preserving protection's ledger-local atomicity.
pub(crate) fn snapshot_placement_candidates(
    registry: &DeviceRegistry,
    request: &PlacementRequest,
    ordinals: impl IntoIterator<Item = usize>,
    protected_ordinal: Option<usize>,
) -> Result<Vec<PlacementCandidateSnapshot>> {
    // Every healthy device participates so the placement loop can distinguish
    // transiently busy devices (capacity fits, equation not yet satisfied)
    // from devices that can never fit the required floor.
    let mut candidates = Vec::with_capacity(registry.len());
    for ordinal in ordinals {
        let context = registry.context_by_ordinal(ordinal)?;
        if !profile_supports_placement_request(
            registry.resource_profile(context.device().ordinal()),
            &request.required_capabilities,
        ) {
            continue;
        }
        let snapshot = context.ledger().snapshot()?;
        if !matches!(snapshot.health, super::DeviceAdmissionHealth::Open) {
            continue;
        }
        let required_floor_bytes = request.required_floor_bytes_for(context.ledger())?;
        let can_ever_fit_required_floor = context.ledger().can_ever_fit_required_floor_bytes(required_floor_bytes)?;
        let max_new_grant_bytes =
            snapshot.max_new_grant_bytes(required_floor_bytes, protected_ordinal == Some(ordinal));
        candidates.push(PlacementCandidateSnapshot {
            candidate: PlacementCandidate {
                ordinal: context.device().ordinal(),
                required_floor_bytes,
                max_new_grant_bytes: max_new_grant_bytes.unwrap_or(0),
                active_attempts: snapshot.active_attempts,
                live_grants_bytes: snapshot.live_grants_bytes,
                worker_depth: snapshot.worker_depth,
                stream_slots_available: snapshot.stream_slots_available(),
            },
            can_admit_now: max_new_grant_bytes.is_some(),
            can_ever_fit_required_floor,
        });
    }
    Ok(candidates)
}

/// Produces the stable direct-placement order from the same snapshots that
/// queue selection uses.  Runnable candidates are tried first; saturated but
/// otherwise valid candidates remain ordered for permanent-capacity analysis.
pub(crate) fn ordered_placement_candidate_snapshots(
    registry: &DeviceRegistry,
    request: &PlacementRequest,
    ordinals: impl IntoIterator<Item = usize>,
    protected_ordinal: Option<usize>,
) -> Result<Vec<PlacementCandidateSnapshot>> {
    let mut candidates = snapshot_placement_candidates(registry, request, ordinals, protected_ordinal)?;
    candidates.sort_by(|left, right| {
        right
            .can_admit_now
            .cmp(&left.can_admit_now)
            .then_with(|| compare_placement_candidates(&left.candidate, &right.candidate))
    });
    Ok(candidates)
}

/// Applies the immutable capability gate before a candidate can be observed
/// by the ledger ranking. Keeping this predicate separate makes it explicit
/// that no capacity or load signal may resurrect an incapable device.
fn profile_supports_placement_request(
    profile: Option<&DeviceResourceProfile>,
    required_capabilities: &[ReservationRequirement],
) -> bool {
    profile.is_some_and(|profile| profile.device_capabilities.supports_all(required_capabilities))
}

/// Places queries on the device with the most spare capacity after admission.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct LeastLoadedPlacement;

impl PlacementPolicy for LeastLoadedPlacement {
    fn candidates(&self, registry: &DeviceRegistry, request: &PlacementRequest) -> Result<Vec<cudf::DeviceId>> {
        let candidates = ordered_placement_candidate_snapshots(
            registry,
            request,
            registry.contexts().map(|context| context.device().ordinal()),
            None,
        )?;
        Ok(candidates
            .into_iter()
            .filter_map(|candidate| {
                registry.context_by_ordinal(candidate.candidate.ordinal).ok().map(|context| context.device())
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        capability::{ContractQuality, ReservationRequirement},
        runtime::{DeviceCapabilities, DeviceCapability, ReservationKind},
    };

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
            required_floor_bytes: 0,
            max_new_grant_bytes: max_new_grant,
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
    fn shared_candidate_selection_uses_the_same_stream_tie_break_for_queue_admission() {
        let selected = preferred_placement_candidate([candidate(7, 100, 1, 50, 2, 3), candidate(4, 100, 1, 50, 2, 4)])
            .expect("candidates are present");

        assert_eq!(selected.ordinal, 4);
    }

    #[test]
    fn capability_gate_excludes_incapable_profiles_before_placement_ranking() {
        let requirement = ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "test_operator",
            "test_operator_contract",
            ContractQuality::CapabilityProof,
        );
        let incapable = DeviceResourceProfile::resolve_with_capabilities(
            0,
            1_000,
            0,
            0,
            100,
            1,
            1,
            2,
            2,
            DeviceCapabilities::only(std::iter::empty::<DeviceCapability>()),
        )
        .unwrap();
        let capable = DeviceResourceProfile::resolve_with_capabilities(
            1,
            1_000,
            0,
            0,
            100,
            1,
            1,
            2,
            2,
            DeviceCapabilities::only([DeviceCapability::new(
                ReservationKind::OperatorOutput,
                "test_operator_contract",
            )]),
        )
        .unwrap();

        assert!(!profile_supports_placement_request(Some(&incapable), &[requirement.clone()]));
        assert!(profile_supports_placement_request(Some(&capable), &[requirement]));
    }
}
