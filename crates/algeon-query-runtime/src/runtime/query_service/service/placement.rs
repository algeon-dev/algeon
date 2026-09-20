//! Device placement and immutable-grant admission against per-device ledgers.

use query_planning::{
    admission::{AttemptRequirement, ExecutionCapability},
    error::{QueryError, QueryErrorCode, Result},
};

use super::{
    super::{
        admission::{DeviceAdmissionEntries, PlacedQuery},
        device_waiters::PendingDeviceSlotNotifications,
        resources::QueryServiceResources,
    },
    QueryService,
};
use crate::{
    observability::{AdmissionRequestId, AttemptId},
    runtime::{
        gpu::{
            DeviceAdmissionHealth, PlacementRequest,
            placement::{
                PlacementCandidateSnapshot, compare_placement_candidates, ordered_placement_candidate_snapshots,
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

impl QueryService {
    /// Folds persistent native CUDA faults into their owning scheduler ledgers.
    ///
    /// Callers own deferred shared-waiter notification: queue maintenance may
    /// hold the admission mutex, while direct placement already owns a pending
    /// notification set. Returning only ordinals whose ledgers transitioned
    /// keeps notification outside those critical sections.
    pub(in crate::runtime::query_service::service) fn reconcile_native_admission_health(&self) -> Result<Vec<usize>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(Vec::new());
        };
        let mut transitioned_ordinals = Vec::new();
        for context in resources.devices.contexts() {
            if context.synchronize_native_admission_health()? {
                transitioned_ordinals.push(context.device().ordinal());
            }
        }
        Ok(transitioned_ordinals)
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
        slot_notifications: &mut PendingDeviceSlotNotifications,
    ) -> Result<Option<PlacedQuery>> {
        let _fleet_admission = self.inner.fleet_admission.lock().map_err(|_error| {
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
        let Some(resources) = &self.inner.resources else {
            let Some(ledger) = &self.inner.fallback_device_ledger else {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.placement",
                    "admission-only service is missing its synthetic device ledger",
                )
                .with_fact("reason", "synthetic_ledger_missing"));
            };
            return ledger.try_admit_for_attempt(admission_request_id, attempt_id, requested_query_cap_bytes).map(
                |grant| grant.map(|memory_grant| PlacedQuery { device_ordinal: ledger.device_ordinal(), memory_grant }),
            );
        };
        let context = resources.devices.context_by_ordinal(device_ordinal)?;
        if context.synchronize_native_admission_health()? {
            slot_notifications.record_device_health_transition(device_ordinal);
            return Ok(None);
        }
        match context.try_admit_attempt(admission_request_id, attempt_id, requested_query_cap_bytes) {
            Ok(Some(memory_grant)) => Ok(Some(PlacedQuery { device_ordinal, memory_grant })),
            Ok(None) => {
                if !matches!(context.ledger().health()?, DeviceAdmissionHealth::Open) {
                    slot_notifications.record_device_health_transition(device_ordinal);
                }
                Ok(None)
            },
            Err(error) => Err(error),
        }
    }

    pub(in crate::runtime::query_service::service) fn placement_candidates_for_entries(
        &self,
        resources: &QueryServiceResources,
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
            placements.extend(ordered_placement_candidate_snapshots(
                &resources.devices,
                &placement_request,
                [ordinal],
                None,
            )?);
        }
        placements.sort_by(|left, right| {
            right
                .can_admit_now
                .cmp(&left.can_admit_now)
                .then_with(|| compare_placement_candidates(&left.candidate, &right.candidate))
        });
        Ok(placements)
    }
}
