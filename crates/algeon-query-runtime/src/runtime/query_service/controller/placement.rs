//! Device placement and immutable-grant admission against per-device ledgers.

use query_planning::{
    admission::{AttemptRequirement, ExecutionCapability, capability_labels},
    error::{QueryError, QueryErrorCode, Result},
};

use super::{
    super::admission::{DeviceAdmissionEntries, PlacedQuery},
    AdmissionController,
};
use crate::{
    observability::{AdmissionRequestId, AttemptId},
    runtime::{
        context::AdmittedPlanIdentities,
        gpu::{
            PlacementRequest,
            placement::{
                PlacementCandidateSnapshot, compare_placement_candidate_snapshots, placement_candidate_snapshot,
            },
        },
        query_service::admission::missing_device_admission_entry_error,
    },
};

/// Returns the immutable capabilities required before placement.
pub(in crate::runtime::query_service) fn required_capabilities_for_placement(
    requirement: &AttemptRequirement,
) -> Vec<ExecutionCapability> {
    requirement.required_capabilities.clone()
}

impl AdmissionController {
    /// Joins a fragment planned after an attempt's claim to the grant placed
    /// on `device_ordinal`. The claim's placement stands: the fragment joins
    /// only when that device provides every capability it requires, and
    /// neither the queue nor the ledger changes.
    pub(in crate::runtime::query_service) fn join_admitted_fragment(
        &self,
        device_ordinal: usize,
        identities: &AdmittedPlanIdentities,
        fragment: &AttemptRequirement,
    ) -> Result<()> {
        if !self.device_supports_capabilities(device_ordinal, &fragment.required_capabilities) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "query_engine.query_service.admitted_fragment",
                "the granted device does not support every native capability the fragment requires",
            )
            .with_fact("reason", "granted_device_capability_unavailable")
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("required_capabilities", capability_labels(&fragment.required_capabilities)));
        }
        identities.join(fragment)
    }

    /// Folds persistent native CUDA faults into their owning scheduler ledgers.
    ///
    /// A ledger that closes notifies this service through its admission
    /// observer, so queued requests recheck their device state.
    pub(in crate::runtime::query_service) fn reconcile_native_admission_health(&self) -> Result<()> {
        self.devices.reconcile_native_health()
    }

    /// Admits the selected waiter on the queue-selected device.
    ///
    /// The queue's oldest-runnable decision already ranked eligible devices;
    /// this performs the health check and the ledger transaction on exactly
    /// that device so overtake accounting
    /// and protection consumption observe the same ordinal.
    pub(in crate::runtime::query_service) fn try_place_on_device(
        &self,
        device_ordinal: usize,
        admission_request_id: AdmissionRequestId,
        attempt_id: AttemptId,
        requirement: &AttemptRequirement,
        requested_query_cap_bytes: Option<u64>,
    ) -> Result<Option<PlacedQuery>> {
        let _fleet_admission = self.fleet_admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.placement",
                "fleet admission gate was poisoned",
            )
            .with_fact("reason", "fleet_admission_gate_poisoned")
        })?;
        let required_capabilities = required_capabilities_for_placement(requirement);
        if !self.device_supports_capabilities(device_ordinal, &required_capabilities) {
            return Err(self.no_capable_device_error(&required_capabilities));
        }
        if self.devices.synchronize_native_health(device_ordinal)? {
            return Ok(None);
        }
        Ok(self
            .devices
            .ledger(device_ordinal)?
            .try_admit_for_attempt(admission_request_id, attempt_id, requested_query_cap_bytes)?
            .map(|memory_grant| PlacedQuery { device_ordinal, memory_grant }))
    }

    pub(in crate::runtime::query_service) fn placement_candidates_for_entries(
        &self,
        device_entries: &DeviceAdmissionEntries,
        ordinals: impl IntoIterator<Item = usize>,
        requested_query_cap_bytes: Option<u64>,
    ) -> Result<Vec<PlacementCandidateSnapshot>> {
        let mut placements = Vec::new();
        for ordinal in ordinals {
            let requirement =
                device_entries.get(&ordinal).ok_or_else(|| missing_device_admission_entry_error(ordinal))?;
            let placement_request = PlacementRequest::new(0)
                .with_requested_query_cap_bytes(requested_query_cap_bytes)
                .with_attempt_requirement(requirement.clone())
                .with_required_capabilities(required_capabilities_for_placement(requirement));
            let ledger = self.ledger_for_ordinal(ordinal)?;
            placements.extend(placement_candidate_snapshot(&ledger, &placement_request)?);
        }
        placements.sort_by(compare_placement_candidate_snapshots);
        Ok(placements)
    }
}
