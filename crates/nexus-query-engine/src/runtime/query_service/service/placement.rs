//! Device placement and immutable-grant admission against per-device ledgers.

use super::*;
use crate::runtime::gpu::placement::ordered_placement_candidate_snapshots;

/// Returns the immutable capabilities required before placement.
pub(in crate::runtime::query_service) fn required_capabilities_for_placement(
    requirement: &AttemptRequirement,
) -> Vec<crate::capability::ReservationRequirement> {
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
        attempt_id: crate::observability::AttemptId,
        requirement: &AttemptRequirement,
        slot_notifications: &mut PendingDeviceSlotNotifications,
    ) -> Result<Option<PlacedQuery>> {
        let _fleet_admission = self.inner.fleet_admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.placement",
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
                    "nexus_query_engine.query_service.placement",
                    "admission-only service is missing its synthetic device ledger",
                )
                .with_fact("reason", "synthetic_ledger_missing"));
            };
            return ledger.try_admit_for_attempt(admission_request_id, attempt_id, requirement).map(|grant| {
                grant.map(|memory_grant| PlacedQuery { device_ordinal: ledger.device_ordinal(), memory_grant })
            });
        };
        let context = resources.devices.context_by_ordinal(device_ordinal)?;
        if context.synchronize_native_admission_health()? {
            slot_notifications.record_device_health_transition(device_ordinal);
            return Ok(None);
        }
        match context.try_admit_attempt(admission_request_id, attempt_id, requirement) {
            Ok(Some(memory_grant)) => Ok(Some(PlacedQuery { device_ordinal, memory_grant })),
            Ok(None) => {
                if !matches!(context.ledger().health()?, crate::runtime::gpu::DeviceAdmissionHealth::Open) {
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
        protected_ordinal: Option<usize>,
    ) -> Result<Vec<crate::runtime::gpu::placement::PlacementCandidateSnapshot>> {
        let mut placements = Vec::new();
        for ordinal in ordinals {
            let entry = device_entries
                .get(&ordinal)
                .ok_or_else(|| super::super::admission::missing_device_admission_entry_error(ordinal))?;
            let requirement = &entry.requirement;
            let placement_request = PlacementRequest::new(0)
                .with_attempt_requirement(requirement.clone())
                .with_required_capabilities(required_capabilities_for_placement(requirement));
            placements.extend(ordered_placement_candidate_snapshots(
                &resources.devices,
                &placement_request,
                [ordinal],
                protected_ordinal.filter(|protected| *protected == ordinal),
            )?);
        }
        placements.sort_by(|left, right| {
            right.can_admit_now.cmp(&left.can_admit_now).then_with(|| {
                crate::runtime::gpu::placement::compare_placement_candidates(&left.candidate, &right.candidate)
            })
        });
        Ok(placements)
    }

    /// Attempts one immutable-grant admission for the queued request.
    ///
    /// Candidates are filtered by capability/health and the normative ledger
    /// equation, then ranked by the documented stable placement order. The
    /// winning device's ledger transaction reserves the attempt slot and the
    /// immutable grant atomically; there is no separate slot to acquire.
    pub(in crate::runtime::query_service) fn try_place_query(
        &self,
        admission_request_id: AdmissionRequestId,
        attempt_id: crate::observability::AttemptId,
        device_entries: &DeviceAdmissionEntries,
        compatible_devices: &[usize],
        slot_notifications: &mut PendingDeviceSlotNotifications,
    ) -> Result<Option<PlacedQuery>> {
        let _fleet_admission = self.inner.fleet_admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.placement",
                "fleet admission gate was poisoned",
            )
            .with_fact("reason", "fleet_admission_gate_poisoned")
        })?;
        for device_ordinal in self.reconcile_native_admission_health()? {
            slot_notifications.record_device_health_transition(device_ordinal);
        }
        let Some(resources) = &self.inner.resources else {
            // Admission-only services deliberately have no CUDA resources. The
            // configured ordinal labels their synthetic ledger and must not be
            // resolved through CUDA merely to admit a handle.
            let fallback_device_ordinal = self.default_device_ordinal().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.placement",
                    "admission-only service has no configured device ordinal",
                )
            })?;
            let requirement = device_entries
                .get(&fallback_device_ordinal)
                .ok_or_else(|| super::super::admission::missing_device_admission_entry_error(fallback_device_ordinal))?
                .requirement
                .clone();
            let required_capabilities = required_capabilities_for_placement(&requirement);
            if !self.device_supports_capabilities(fallback_device_ordinal, &required_capabilities) {
                return Err(self.no_capable_device_error(&required_capabilities));
            }
            if let Some(error) = self.all_compatible_devices_closed_error(&[fallback_device_ordinal])? {
                return Err(error);
            }
            let Some(ledger) = &self.inner.fallback_device_ledger else {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.placement",
                    "admission-only service is missing its synthetic device ledger",
                )
                .with_fact("reason", "synthetic_ledger_missing"));
            };
            return ledger.try_admit_for_attempt(admission_request_id, attempt_id, &requirement).map(|grant| {
                grant.map(|memory_grant| PlacedQuery { device_ordinal: ledger.device_ordinal(), memory_grant })
            });
        };

        if compatible_devices.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedPlanShape,
                "nexus_query_engine.query_service.placement",
                "no device-specific admission entry is available for placement",
            )
            .with_fact("reason", "missing_device_specific_admission_resolution"));
        }
        // An unknown-isolation cleanup fault is represented by closed device
        // ledgers, not a service-side latch.  Check that canonical controller
        // state before treating a closed fleet as a capacity-floor error.
        if let Some(error) = self.all_compatible_devices_closed_error(&compatible_devices)? {
            return Err(error);
        }

        let mut any_device_can_fit_floor = false;
        let mut max_candidate_required_floor_bytes = 0u64;
        let placements =
            self.placement_candidates_for_entries(resources, device_entries, compatible_devices.iter().copied(), None)?;
        #[cfg(test)]
        if let Some(pause) = take_direct_placement_snapshot_pause(self) {
            pause.wait();
        }
        for placement in placements {
            let candidate = placement.candidate;
            let context = resources.devices.context_by_ordinal(candidate.ordinal)?;
            if context.synchronize_native_admission_health()? {
                slot_notifications.record_device_health_transition(candidate.ordinal);
                continue;
            }
            max_candidate_required_floor_bytes = max_candidate_required_floor_bytes.max(candidate.required_floor_bytes);
            if placement.can_ever_fit_required_floor {
                any_device_can_fit_floor = true;
            }
            // Ranking and permanent-capacity classification come from the
            // same ledger snapshot as queue selection. The later ledger
            // transaction is still authoritative; a stale affirmative
            // snapshot may simply fail to reserve and leave the request
            // queued for the next canonical scan.
            if !placement.can_admit_now {
                continue;
            }
            let requirement = &device_entries
                .get(&candidate.ordinal)
                .ok_or_else(|| super::super::admission::missing_device_admission_entry_error(candidate.ordinal))?
                .requirement;
            match context.try_admit_attempt(admission_request_id, attempt_id, requirement) {
                Ok(Some(memory_grant)) => {
                    return Ok(Some(PlacedQuery { device_ordinal: candidate.ordinal, memory_grant }));
                },
                Ok(None) => {
                    if !matches!(context.ledger().health()?, crate::runtime::gpu::DeviceAdmissionHealth::Open) {
                        slot_notifications.record_device_health_transition(candidate.ordinal);
                    }
                },
                Err(error) => return Err(error),
            }
        }
        // A native fault can close the final ledger after the initial fleet
        // check or after its ranking snapshot. Health is ledger-owned, so
        // re-read the canonical state before calling a remaining open fleet a
        // capacity failure.
        if let Some(error) = self.all_compatible_devices_closed_error(&compatible_devices)? {
            return Err(error);
        }
        if any_device_can_fit_floor {
            return Ok(None);
        }
        Err(QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.placement",
            "no configured CUDA device has enough capacity for the attempt required floor",
        )
        .with_fact("max_candidate_required_floor_bytes", max_candidate_required_floor_bytes)
        .with_fact("reason", "required_floor_exceeds_device_capacity"))
    }
}

/// Test-only rendezvous between direct candidate capture and its authoritative
/// ledger transaction. It proves that a fleet health transition in this gap is
/// reported from the ledger, not classified as a static capacity failure.
#[cfg(test)]
#[derive(Debug)]
struct DirectPlacementSnapshotPause {
    entered: std::sync::mpsc::SyncSender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

#[cfg(test)]
impl DirectPlacementSnapshotPause {
    fn wait(&self) {
        self.entered.send(()).expect("direct placement snapshot pause receiver remains live");
        self.release
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv()
            .expect("direct placement snapshot pause is released");
    }
}

#[cfg(test)]
#[derive(Debug)]
struct DirectPlacementSnapshotPauseRegistration {
    service: std::sync::Weak<QueryServiceInner>,
    pause: Arc<DirectPlacementSnapshotPause>,
}

#[cfg(test)]
static DIRECT_PLACEMENT_SNAPSHOT_PAUSE: std::sync::OnceLock<Mutex<Vec<DirectPlacementSnapshotPauseRegistration>>> =
    std::sync::OnceLock::new();

#[cfg(test)]
fn direct_placement_snapshot_pause_registry() -> &'static Mutex<Vec<DirectPlacementSnapshotPauseRegistration>> {
    DIRECT_PLACEMENT_SNAPSHOT_PAUSE.get_or_init(|| Mutex::new(Vec::new()))
}

#[cfg(test)]
fn take_direct_placement_snapshot_pause(service: &QueryService) -> Option<Arc<DirectPlacementSnapshotPause>> {
    let mut registrations =
        direct_placement_snapshot_pause_registry().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    registrations.retain(|registered| registered.service.upgrade().is_some());
    registrations
        .iter()
        .position(|registered| registered.service.upgrade().is_some_and(|target| Arc::ptr_eq(&target, &service.inner)))
        .map(|position| registrations.remove(position).pause)
}

#[cfg(test)]
impl QueryService {
    pub(in crate::runtime::query_service) fn pause_next_direct_placement_after_snapshot_for_test(
        &self,
    ) -> TestPauseHandle {
        let (entered_sender, entered_receiver) = std::sync::mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
        let pause =
            Arc::new(DirectPlacementSnapshotPause { entered: entered_sender, release: Mutex::new(release_receiver) });
        let service = Arc::downgrade(&self.inner);
        let mut registrations =
            direct_placement_snapshot_pause_registry().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        registrations.retain(|registered| registered.service.upgrade().is_some());
        assert!(
            !registrations.iter().any(|registered| registered.service.ptr_eq(&service)),
            "only one direct placement snapshot pause may be armed for a service at a time"
        );
        registrations
            .push(DirectPlacementSnapshotPauseRegistration { service: service.clone(), pause: Arc::clone(&pause) });
        TestPauseHandle::new("direct placement snapshot", entered_receiver, release_sender).with_drop_cleanup(
            move || {
                let mut registrations = direct_placement_snapshot_pause_registry()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(position) = registrations.iter().position(|registered| {
                    registered.service.ptr_eq(&service) && Arc::ptr_eq(&registered.pause, &pause)
                }) {
                    registrations.remove(position);
                }
            },
        )
    }
}
