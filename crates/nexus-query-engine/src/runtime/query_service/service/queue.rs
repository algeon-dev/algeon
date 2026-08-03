//! Admission queue selection, sweep, and lock helpers.
//!
//! Queue order is strictly oldest-first by submission sequence. The first
//! runnable waiter wins; a waiter blocked on one device never idles another
//! compatible device for later work. There is no tenant rotation, priority,
//! weight, or QoS in queue state. (Finite-overtake liveness protection lands
//! with the P2.4 controller state machine.)

use super::*;
use crate::runtime::gpu::placement::{
    PlacementCandidate, ordered_placement_candidate_snapshots, preferred_placement_candidate,
};

/// Returns the only ledgers on which a waiter may be placed right now.
///
/// A slot protection is ledger-local. Until it is consumed or explicitly
/// retargeted, placement on any other compatible ledger would orphan the
/// protected capacity.
fn placement_candidate_ordinals(
    compatible_devices: &[usize],
    protected_device: Option<usize>,
) -> impl Iterator<Item = usize> + '_ {
    compatible_devices
        .iter()
        .copied()
        .filter(move |ordinal| protected_device.is_none_or(|protected| protected == *ordinal))
}

/// Result of atomically trying the best ledger-local protection target.
///
/// Ranking is only an observation. `Installed` is returned solely after the
/// target ledger accepts the protection under its own lock; contention and a
/// health race remain retryable queue states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExhaustedWaiterProtection {
    Installed(usize),
    Contended,
    NoHealthyTarget,
}

impl QueryService {
    pub(in crate::runtime::query_service) fn remove_queued_request_without_admission_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        let removed = remove_queued_request(state, admission_request_id);
        if let Some(request) = &removed
            && let Err(error) = self.end_request_protection(request)
        {
            eprintln!("[nexus-query-engine] warning: failed to end queue protection: {error}");
        }
        removed
    }

    /// Computes the immutable capability-compatible device set at submission.
    ///
    /// This intentionally precedes queue insertion: a requirement with no
    /// capable installed device is a permanent local rejection, never a
    /// capacity waiter. Health and ledger capacity are evaluated afterwards
    /// while selecting among this frozen set.
    pub(in crate::runtime::query_service) fn compatible_devices_for_capabilities(
        &self,
        required_capabilities: &[crate::capability::ReservationRequirement],
    ) -> Result<Vec<usize>> {
        let mut devices = match &self.inner.resources {
            Some(resources) => resources
                .devices
                .device_ordinals()
                .iter()
                .copied()
                .filter(|ordinal| resources.devices.supports_capabilities(*ordinal, required_capabilities))
                .collect(),
            None => self
                .inner
                .fallback_device_ledger
                .as_ref()
                .filter(|ledger| ledger.profile().device_capabilities.supports_all(required_capabilities))
                .map(|ledger| vec![ledger.device_ordinal()])
                .unwrap_or_default(),
        };
        devices.sort_unstable();
        devices.dedup();
        Ok(devices)
    }

    /// Resolves the request's immutable requirement without allowing a missing
    /// capability proof to pass through an exact device profile as an empty set.
    pub(in crate::runtime::query_service) fn resolve_attempt_requirement_for_submission(
        &self,
        attempt_requirement: Option<AttemptRequirement>,
    ) -> Result<AttemptRequirement> {
        if let Some(requirement) = attempt_requirement {
            return Ok(requirement);
        }
        let restricted_device_count = match &self.inner.resources {
            Some(resources) => resources
                .devices
                .contexts()
                .filter(|context| {
                    context.ledger().profile().device_capabilities.requires_explicit_attempt_requirement()
                })
                .count(),
            None => self
                .inner
                .fallback_device_ledger
                .as_ref()
                .is_some_and(|ledger| ledger.profile().device_capabilities.requires_explicit_attempt_requirement())
                .into(),
        };
        if restricted_device_count == 0 {
            return Ok(AttemptRequirement::empty());
        }
        Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.query_service.admission",
            "an explicit attempt requirement is required by the installed device capability profiles",
        )
        .with_fact("reason", "explicit_attempt_requirement_required")
        .with_fact("restricted_device_count", restricted_device_count))
    }

    /// Checks one selected ordinal against the profile frozen at installation.
    /// This defensive gate preserves the contract for direct placement calls
    /// and fallback queue drives as well as the normal compatible-set path.
    pub(in crate::runtime::query_service) fn device_supports_capabilities(
        &self,
        device_ordinal: usize,
        required_capabilities: &[crate::capability::ReservationRequirement],
    ) -> bool {
        match &self.inner.resources {
            Some(resources) => resources.devices.supports_capabilities(device_ordinal, required_capabilities),
            None if self.default_device_ordinal() == Some(device_ordinal) => self
                .inner
                .fallback_device_ledger
                .as_ref()
                .is_some_and(|ledger| ledger.profile().device_capabilities.supports_all(required_capabilities)),
            None => false,
        }
    }

    /// Returns whether an ordinal can accept a queued placement right now.
    ///
    /// The device ledger is the sole scheduler health authority. Native CUDA
    /// samples are diagnostics and cannot race ledger admission decisions.
    fn is_device_admission_healthy(&self, device_ordinal: usize) -> Result<bool> {
        let ledger = self.ledger_for_ordinal(device_ordinal)?;
        Ok(matches!(ledger.health()?, crate::runtime::gpu::DeviceAdmissionHealth::Open))
    }

    /// Returns a terminal health error only when every capability-compatible
    /// device has permanently closed admission.  A healthy but capacity-bound
    /// device must remain queued; a waiter must never turn a temporary ledger
    /// shortage into a false device failure.  Conversely, once every candidate
    /// is quarantined there is no release that can make this request runnable,
    /// so retaining it until `max_wait` only hides the actionable CUDA fault.
    pub(in crate::runtime::query_service) fn all_compatible_devices_closed_error(
        &self,
        compatible_devices: &[usize],
    ) -> Result<Option<QueryError>> {
        let mut first_closed_ledger = None;
        let mut all_closed_for_unknown_isolation = true;
        for ordinal in compatible_devices {
            let ledger = self.ledger_for_ordinal(*ordinal)?;
            match ledger.health()? {
                crate::runtime::gpu::DeviceAdmissionHealth::Open => {
                    return Ok(None);
                },
                crate::runtime::gpu::DeviceAdmissionHealth::Closed { reason, generation } => {
                    all_closed_for_unknown_isolation &= reason.starts_with("unknown-isolation backend failure:");
                    first_closed_ledger.get_or_insert((*ordinal, reason, generation));
                },
            }
        }

        if let Some((device_ordinal, detail, generation)) = first_closed_ledger {
            let code = if self.inner.resources.is_some() {
                QueryErrorCode::CudfPersistentFailure
            } else {
                QueryErrorCode::ServiceOverloaded
            };
            let mut error = QueryError::new(
                code,
                "nexus_query_engine.query_service.admission",
                "every compatible device has closed admission",
            )
            .with_fact(
                "reason",
                if all_closed_for_unknown_isolation { "backend_unhealthy" } else { "no_healthy_compatible_device" },
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("device_admission_health", "closed")
            .with_fact("device_health_generation", generation)
            .with_fact("device_health_detail", detail.clone());
            if all_closed_for_unknown_isolation {
                let detail = detail.strip_prefix("unknown-isolation backend failure: ").unwrap_or(detail.as_str());
                error = error.with_fact("detail", detail);
            }
            return Ok(Some(error));
        }
        Ok(None)
    }

    pub(in crate::runtime::query_service) fn no_capable_device_error(
        &self,
        required_capabilities: &[crate::capability::ReservationRequirement],
    ) -> QueryError {
        QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.query_service.admission",
            "no installed device supports every required native capability",
        )
        .with_fact("reason", "required_device_capability_unavailable")
        .with_fact("required_capabilities", AttemptRequirement::capability_labels(required_capabilities))
        .with_fact(
            "configured_device_count",
            self.inner.resources.as_ref().map_or(1, |resources| resources.devices.len()),
        )
        .with_fact("compatible_device_count", 0)
    }

    pub(in crate::runtime::query_service) fn reject_impossible_requirement(
        &self,
        requirement: &AttemptRequirement,
        required_capabilities: &[crate::capability::ReservationRequirement],
        compatible_devices: &[usize],
    ) -> Result<()> {
        if compatible_devices.is_empty() && !required_capabilities.is_empty() {
            return Err(self.no_capable_device_error(required_capabilities));
        }
        let mut maximum = None;
        let mut can_ever_fit = false;
        for ordinal in compatible_devices {
            let ledger = self.ledger_for_ordinal(*ordinal)?;
            let managed = ledger.managed_capacity_bytes();
            maximum = Some(maximum.unwrap_or(0u64).max(managed));
            let required_floor = ledger.required_floor_bytes_for(requirement)?;
            can_ever_fit |= ledger.can_ever_fit_required_floor_bytes(required_floor)?;
        }
        if compatible_devices.is_empty() || !can_ever_fit {
            return Err(QueryError::new(
                QueryErrorCode::ServiceOverloaded,
                "nexus_query_engine.query_service.admission",
                "no compatible device can satisfy the attempt required floor",
            )
            .with_fact("reason", "impossible_requirement")
            .with_fact("max_compatible_managed_capacity_bytes", maximum.unwrap_or(0)));
        }
        Ok(())
    }

    pub(in crate::runtime::query_service::service) fn end_request_protection(
        &self,
        request: &AdmissionRequest,
    ) -> Result<()> {
        if let Some(ordinal) = request.protected_device {
            self.ledger_for_ordinal(ordinal)?.end_protection(request.admission_request_id)?;
        }
        Ok(())
    }

    pub(in crate::runtime::query_service) fn ledger_for_ordinal(&self, ordinal: usize) -> Result<Arc<DeviceLedger>> {
        match &self.inner.resources {
            Some(resources) => Ok(resources.devices.context_by_ordinal(ordinal)?.ledger().clone()),
            None if self.default_device_ordinal() == Some(ordinal) => {
                self.inner.fallback_device_ledger.clone().map(Arc::new).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "nexus_query_engine.query_service.queue",
                        "synthetic ledger missing",
                    )
                })
            },
            None => Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.queue",
                "unknown compatible device",
            )),
        }
    }

    pub(in crate::runtime::query_service) fn abandon_queued_request_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        let removed = self.remove_queued_request_without_admission_locked(state, admission_request_id);
        if removed.is_some() {
            self.inner.notify_admission_available();
        }
        removed
    }

    pub(in crate::runtime::query_service) fn lock_admission(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        self.inner.admission.lock().map_err(|_error| admission_mutex_poisoned_error())
    }

    /// Removes and terminalizes one queued request after recovering a poisoned
    /// admission guard. The terminal callback runs only after the guard is
    /// released, because it may synchronously reenter backend shutdown.
    pub(in crate::runtime::query_service) fn recover_queued_request_after_admission_poison(
        &self,
        mut state: std::sync::MutexGuard<'_, AdmissionState>,
        admission_request_id: AdmissionRequestId,
    ) -> QueryError {
        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
        let queue_depth = state.queued.len();
        drop(state);

        let error = admission_mutex_poisoned_error();
        if let Some(request) = removed {
            record_admission_lifecycle(
                &request.observation,
                request.admission_request_id,
                admission_rejection(&error),
                queue_depth,
                None,
                None,
            );
            finish_removed_request_error(Some(request), &error);
        }
        error
    }

    pub(in crate::runtime::query_service) fn admission_is_closed(&self, state: &AdmissionState) -> bool {
        !self.inner.admission_open.load(Ordering::Acquire) || state.closed
    }

    /// Terminal admission rejection for service shutdown. Persistent device
    /// health is evaluated against each request's compatible ledgers, because
    /// that is the scheduler's sole health authority.
    pub(in crate::runtime::query_service) fn admission_terminal_rejection(
        &self,
        state: &AdmissionState,
    ) -> Option<QueryError> {
        if self.admission_is_closed(state) {
            return Some(service_closed_error());
        }
        None
    }

    pub(in crate::runtime::query_service::service) fn lock_admission_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        match crate::runtime::deadline::mutex_lock_until(&self.inner.admission, deadline) {
            Ok(state) => Ok(state),
            Err(crate::runtime::deadline::DeadlineMutexLockError::Poisoned) => self.lock_admission(),
            Err(crate::runtime::deadline::DeadlineMutexLockError::TimedOut) => {
                Err(query_service_shutdown_timeout(operation, None, "admission_mutex"))
            },
        }
    }

    pub(in crate::runtime::query_service) fn sweep_expired_queued_requests_locked(
        &self,
        state: &mut AdmissionState,
    ) -> QueueSweep {
        let transitioned_device_ordinals = match self.reconcile_native_admission_health() {
            Ok(transitioned_device_ordinals) => transitioned_device_ordinals,
            Err(error) => {
                eprintln!("[nexus-query-engine] warning: native admission-health reconciliation failed: {error}");
                Vec::new()
            },
        };
        // Terminal intent has a strict precedence. A caller that cancelled or
        // expired while queued must be removed before a concurrent health
        // transition can retarget its protection and replace that terminal
        // result with a persistent-device failure.
        let now = Instant::now();
        let mut expired = Vec::new();
        let mut cancelled = Vec::new();
        let mut retained = VecDeque::with_capacity(state.queued.len());
        while let Some(request) = state.queued.pop_front() {
            if request.execution_cancellation.is_cancelled() {
                let _ = self.end_request_protection(&request);
                let _ = request.terminal.record(admission_cancellation_error());
                cancelled.push(request);
            } else if request.deadline <= now {
                let _ = self.end_request_protection(&request);
                let disposition = request.terminal.record(service_overloaded_error(
                    "queue_wait_elapsed",
                    state.queued.len(),
                    self.inner.queue_bounds.max_waiters,
                ));
                expired.push((request, disposition));
            } else {
                retained.push_back(request);
            }
        }
        state.queued = retained;

        // Retarget only live waiters. It may remove protected waiters whose
        // last capable device closed, after which the ordinary health sweep
        // handles unprotected waiters with the same disposition.
        let mut terminated = match self.retarget_protections_locked(state) {
            Ok(terminated) => terminated,
            Err(error) => {
                eprintln!("[nexus-query-engine] warning: protection retarget failed: {error}");
                Vec::new()
            },
        };
        let mut retained = VecDeque::with_capacity(state.queued.len());
        while let Some(request) = state.queued.pop_front() {
            match self.all_compatible_devices_closed_error(&request.compatible_devices) {
                Ok(Some(error)) => {
                    let _ = self.end_request_protection(&request);
                    terminated.push((request, error));
                },
                Ok(None) => retained.push_back(request),
                Err(error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to evaluate queued-device health; retaining waiter: {error}"
                    );
                    retained.push_back(request);
                },
            }
        }
        state.queued = retained;
        if !expired.is_empty() {
            self.inner.notify_admission_available();
            for (request, _) in &expired {
                record_admission_lifecycle(
                    &request.observation,
                    request.admission_request_id,
                    AdmissionLifecycleState::TimedOut,
                    state.queued.len(),
                    None,
                    None,
                );
            }
        }
        if !cancelled.is_empty() {
            self.inner.notify_admission_available();
            for request in &cancelled {
                record_admission_lifecycle(
                    &request.observation,
                    request.admission_request_id,
                    AdmissionLifecycleState::CancellationRequested,
                    state.queued.len(),
                    None,
                    None,
                );
                record_admission_lifecycle(
                    &request.observation,
                    request.admission_request_id,
                    AdmissionLifecycleState::Cancelled,
                    state.queued.len(),
                    None,
                    None,
                );
            }
        }
        if !terminated.is_empty() {
            self.inner.notify_admission_available();
        }
        let mut terminated_requests = Vec::with_capacity(terminated.len());
        for (request, error) in terminated {
            let rejection = admission_rejection(&error);
            let disposition = request.terminal.record(error);
            terminated_requests.push((request, disposition, rejection));
        }
        QueueSweep {
            expired,
            cancelled,
            terminated: terminated_requests,
            transitioned_device_ordinals,
            #[cfg(test)]
            finish_pause: take_queue_sweep_finish_pause(self),
        }
    }

    /// Selects the oldest runnable waiter and its best eligible device.
    ///
    /// Each waiter's eligible devices are ranked by the documented stable
    /// placement order (spare after admission, active attempts, live grants,
    /// worker depth, stream availability, ordinal). A waiter blocked on every
    /// compatible device is skipped so later work keeps independent devices
    /// busy.
    pub(in crate::runtime::query_service) fn select_next_runnable_locked(
        &self,
        state: &mut AdmissionState,
    ) -> Result<Option<(usize, usize)>> {
        // A protected waiter can leave the queue (for example, through
        // cancellation) while a younger waiter that already exhausted its
        // allowance remains. Reinstall that protection before looking for a
        // runnable placement, not after a younger request has consumed the
        // next slot. This keeps the finite-overtake bound exact.
        let previous_protections = state.queued.iter().map(|request| request.protected_device).collect::<Vec<_>>();
        if let Err(error) = self.restore_exhausted_overtake_protections_locked(state) {
            return Err(self.rollback_new_exhausted_overtake_protections(state, &previous_protections, error));
        }
        for (position, request) in state.queued.iter().enumerate() {
            let Some(best) = self.best_eligible_device_for(request)? else {
                continue;
            };
            return Ok(Some((position, best)));
        }
        Ok(None)
    }

    /// Ranks one waiter's currently eligible devices and returns the best.
    ///
    /// A protection belongs to one ledger. Only that ledger can atomically
    /// replace the protected slot and liveness bytes with this waiter's grant,
    /// so a protected waiter must not be placed on another compatible device.
    /// Health-driven retargeting releases and recreates the protection before
    /// selection considers another ordinal.
    pub(in crate::runtime::query_service::service) fn best_eligible_device_for(
        &self,
        request: &AdmissionRequest,
    ) -> Result<Option<usize>> {
        if let Some(resources) = &self.inner.resources {
            let placement_request = PlacementRequest::new(0)
                .with_attempt_requirement(request.requirement.clone())
                .with_required_capabilities(super::placement::required_capabilities_for_placement(
                    &request.requirement,
                ));
            let candidates = ordered_placement_candidate_snapshots(
                &resources.devices,
                &placement_request,
                placement_candidate_ordinals(&request.compatible_devices, request.protected_device),
                request.protected_device,
            )?;
            return Ok(candidates
                .into_iter()
                .find(|candidate| candidate.can_admit_now)
                .map(|candidate| candidate.candidate.ordinal));
        }

        // Admission-only fixtures have one explicit synthetic ledger and no
        // registry to snapshot.  They retain the same ledger-local protected
        // reservation semantics while production queue/direct placement share
        // the registry helper above.
        let mut candidates = Vec::new();
        for ordinal in placement_candidate_ordinals(&request.compatible_devices, request.protected_device) {
            let ledger = self.ledger_for_ordinal(ordinal)?;
            if !self.is_device_admission_healthy(ordinal)? {
                continue;
            }
            let required_floor = ledger.required_floor_bytes_for(&request.requirement)?;
            // One snapshot contains every ranking input. The protected-waiter
            // exception is a property of this queued request, so it is safe to
            // resolve it against the same versioned ledger read.
            let snapshot = ledger.snapshot()?;
            if !matches!(snapshot.health, crate::runtime::gpu::DeviceAdmissionHealth::Open) {
                continue;
            }
            let is_protected_waiter = request.protected_device == Some(ordinal);
            let Some(max_new_grant_bytes) = snapshot.max_new_grant_bytes(required_floor, is_protected_waiter) else {
                continue;
            };
            candidates.push(PlacementCandidate {
                ordinal,
                required_floor_bytes: required_floor,
                max_new_grant_bytes,
                active_attempts: snapshot.active_attempts,
                live_grants_bytes: snapshot.live_grants_bytes,
                worker_depth: snapshot.worker_depth,
                stream_slots_available: snapshot.stream_slots_available(),
            });
        }
        Ok(preferred_placement_candidate(candidates).map(|candidate| candidate.ordinal))
    }

    /// When no waiter is runnable, the oldest waiter still drives the
    /// placement attempt so its memory-blocked wait keeps the
    /// `memory_grant_wait_timeout` taxonomy instead of the plain queue
    /// timeout. The drive fails placement and falls back to waiting.
    pub(in crate::runtime::query_service) fn fallback_driver_selection_locked(
        &self,
        state: &AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Result<Option<(usize, usize)>> {
        // The fallback drive exists only to preserve the memory-blocked
        // timeout taxonomy. The selected device ledger owns the only active
        // attempt limit, so this path never consults a global controller cap.
        let Some(first) = state.queued.front() else {
            return Ok(None);
        };
        if first.admission_request_id != admission_request_id {
            return Ok(None);
        }
        if let Some(resources) = &self.inner.resources {
            let placement_request = PlacementRequest::new(0)
                .with_attempt_requirement(first.requirement.clone())
                .with_required_capabilities(super::placement::required_capabilities_for_placement(&first.requirement));
            // Preserve the same registry ranking as normal queue/direct
            // placement.  Unlike ordinary selection the fallback deliberately
            // drives the best healthy candidate even if it is saturated, so
            // the resulting wait retains the memory-grant timeout taxonomy.
            let candidates = ordered_placement_candidate_snapshots(
                &resources.devices,
                &placement_request,
                placement_candidate_ordinals(&first.compatible_devices, first.protected_device),
                first.protected_device,
            )?;
            return Ok(candidates.into_iter().next().map(|candidate| (0, candidate.candidate.ordinal)));
        }
        for device in placement_candidate_ordinals(&first.compatible_devices, first.protected_device) {
            if self.is_device_admission_healthy(device)? {
                return Ok(Some((0, device)));
            }
        }
        Ok(None)
    }

    /// Ends protections on devices that closed admission and retargets each
    /// affected waiter to its next compatible unprotected device by the
    /// documented `(capacity deficit, active count, ordinal)` order. Waiters
    /// with no remaining compatible unprotected device are terminated.
    pub(in crate::runtime::query_service) fn retarget_protections_locked(
        &self,
        state: &mut AdmissionState,
    ) -> Result<Vec<(AdmissionRequest, QueryError)>> {
        let mut terminated = Vec::new();
        for position in (0..state.queued.len()).rev() {
            let Some(protected_ordinal) = state.queued[position].protected_device else {
                continue;
            };
            let protected_ledger = self.ledger_for_ordinal(protected_ordinal)?;
            if self.is_device_admission_healthy(protected_ordinal)? {
                continue;
            }
            let request_id = state.queued[position].admission_request_id;
            protected_ledger.end_protection(request_id)?;
            state.queued[position].protected_device = None;
            let requirement = state.queued[position].requirement.clone();
            let compatible_devices = state.queued[position].compatible_devices.clone();
            let protection = match self.try_install_exhausted_waiter_protection_after_old_target_release(
                request_id,
                &requirement,
                &compatible_devices,
                Some(protected_ordinal),
            ) {
                Ok(protection) => protection,
                Err(error) => {
                    // The old ledger-local reservation has already ended.
                    // This waiter owns any failure to establish a replacement;
                    // retaining it unprotected would let a later caller fail
                    // while driving queue repair and would violate finite
                    // overtake protection.
                    let removed = state.queued.remove(position).expect("scanned position is valid");
                    terminated.push((removed, error));
                    continue;
                },
            };
            if let ExhaustedWaiterProtection::Installed(ordinal) = protection {
                state.queued[position].protected_device = Some(ordinal);
            }
            // An older waiter may already protect the only viable device.
            // Leave this waiter exhausted but unprotected and retry after a
            // later admission/release; it is not terminally impossible yet.
            if matches!(protection, ExhaustedWaiterProtection::NoHealthyTarget) {
                // Health closure is terminal, not a transient resource
                // shortage. Reuse the queue-wide decision so a protected
                // waiter preserves the same persistent-fault taxonomy as a
                // non-protected waiter on the same device set.
                let error = match self.all_compatible_devices_closed_error(&compatible_devices) {
                    Ok(Some(error)) => error,
                    Ok(None) => service_overloaded_error(
                        "protected_waiter_no_runnable_device",
                        state.queued.len(),
                        self.inner.queue_bounds.max_waiters,
                    )
                    .with_fact("protected_device_ordinal", protected_ordinal),
                    Err(error) => error,
                };
                let removed = state.queued.remove(position).expect("scanned position is valid");
                terminated.push((removed, error));
            }
        }
        Ok(terminated)
    }

    /// Retargets only after the prior ledger-local protection has ended.
    /// Production errors come from the candidate scan or target ledger; the
    /// test-only hook covers that same ownership boundary deterministically.
    fn try_install_exhausted_waiter_protection_after_old_target_release(
        &self,
        request_id: AdmissionRequestId,
        requirement: &AttemptRequirement,
        compatible_devices: &[usize],
        excluded_ordinal: Option<usize>,
    ) -> Result<ExhaustedWaiterProtection> {
        #[cfg(test)]
        if self.inner.protection_retarget_failure_after_release.swap(false, Ordering::AcqRel) {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.queue",
                "injected protection retarget failure after previous target release",
            )
            .with_fact("reason", "injected_protection_retarget_failure"));
        }
        self.try_install_exhausted_waiter_protection(request_id, requirement, compatible_devices, excluded_ordinal)
    }

    /// Ranks every currently viable protection target from one ledger
    /// snapshot, then asks the selected ledgers to atomically install the
    /// reservation. Both retarget and restore use this path so an internal
    /// ledger failure cannot silently turn an exhausted waiter into an
    /// unprotected waiter.
    fn try_install_exhausted_waiter_protection(
        &self,
        request_id: AdmissionRequestId,
        requirement: &AttemptRequirement,
        compatible_devices: &[usize],
        excluded_ordinal: Option<usize>,
    ) -> Result<ExhaustedWaiterProtection> {
        let mut candidates = Vec::new();
        let mut healthy_target_remains = false;
        for ordinal in compatible_devices.iter().copied().filter(|ordinal| Some(*ordinal) != excluded_ordinal) {
            let ledger = self.ledger_for_ordinal(ordinal)?;
            let required_floor = ledger.required_floor_bytes_for(requirement)?;
            if !ledger.can_ever_fit_required_floor_bytes(required_floor)? {
                continue;
            }
            let snapshot = ledger.snapshot()?;
            if !matches!(snapshot.health, crate::runtime::gpu::DeviceAdmissionHealth::Open) {
                continue;
            }
            healthy_target_remains = true;
            if snapshot.protected_slot {
                continue;
            }
            let max_new = snapshot.max_new_grant_bytes(required_floor, false).unwrap_or(0);
            candidates.push((
                required_floor.saturating_sub(max_new),
                snapshot.active_attempts,
                ordinal,
                required_floor,
            ));
        }
        candidates.sort_unstable();

        for (_, _, ordinal, required_floor) in candidates {
            let ledger = self.ledger_for_ordinal(ordinal)?;
            match ledger.protect_slot(request_id, required_floor) {
                Ok(true) => return Ok(ExhaustedWaiterProtection::Installed(ordinal)),
                // Another exhausted waiter won the ledger-local transaction.
                Ok(false) => {},
                // The device closed after the snapshot; a future canonical
                // sweep will retarget or terminalize this waiter.
                Err(error) if error.fact_value("reason") == Some("closed_protection_target") => {},
                Err(error) => return Err(error),
            }
        }
        Ok(if healthy_target_remains {
            ExhaustedWaiterProtection::Contended
        } else {
            ExhaustedWaiterProtection::NoHealthyTarget
        })
    }

    /// Reinstalls a liveness protection for each exhausted waiter that lost
    /// its previous target. Callers hold the admission mutex, so no placement
    /// can consume the newly available slot between this reconciliation and
    /// candidate selection.
    fn restore_exhausted_overtake_protections_locked(&self, state: &mut AdmissionState) -> Result<()> {
        for position in 0..state.queued.len() {
            let (request_id, requirement, compatible_devices) = {
                let request = &state.queued[position];
                if request.overtakes_remaining != 0 || request.protected_device.is_some() {
                    continue;
                }
                (request.admission_request_id, request.requirement.clone(), request.compatible_devices.clone())
            };
            if let ExhaustedWaiterProtection::Installed(ordinal) =
                self.try_install_exhausted_waiter_protection(request_id, &requirement, &compatible_devices, None)?
            {
                state.queued[position].protected_device = Some(ordinal);
            }
        }
        Ok(())
    }

    /// Ends every protection installed by a failed restoration pass and
    /// restores the queue's prior protection references. A restoration can
    /// touch several ledgers, so a later hard failure must not leave an
    /// earlier waiter's slot and liveness bytes withheld indefinitely.
    fn rollback_new_exhausted_overtake_protections(
        &self,
        state: &mut AdmissionState,
        previous_protections: &[Option<usize>],
        mut error: QueryError,
    ) -> QueryError {
        let mut protection_rolled_back = vec![true; state.queued.len()];
        for (position, request) in state.queued.iter().enumerate() {
            if previous_protections[position].is_some() || request.protected_device.is_none() {
                continue;
            }
            let protected_device = request.protected_device.expect("new protection exists");
            let rollback = self
                .ledger_for_ordinal(protected_device)
                .and_then(|ledger| ledger.end_protection(request.admission_request_id).map(|_| ()));
            if let Err(rollback_error) = rollback {
                protection_rolled_back[position] = false;
                error = error.with_fact("overtake_rollback_error", rollback_error.to_string());
            }
        }
        for (position, request) in state.queued.iter_mut().enumerate() {
            if previous_protections[position].is_some() || protection_rolled_back[position] {
                request.protected_device = previous_protections[position];
            }
        }
        error
    }

    /// Accounts for a published later admission and installs protection for
    /// every older still-queued compatible waiter that has exhausted its finite
    /// overtake allowance. Any protection failure rolls back both decrements
    /// and protections created by this call.
    pub(in crate::runtime::query_service) fn record_overtake_locked(
        &self,
        state: &mut AdmissionState,
        admitted_submission_seq: u64,
        device_ordinal: usize,
        publication_deadline: Instant,
    ) -> Result<()> {
        let previous = state
            .queued
            .iter()
            .map(|request| (request.overtakes_remaining, request.protected_device))
            .collect::<Vec<_>>();
        for request in &mut state.queued {
            // A published later admission consumes one overtake only when the
            // older waiter remains queued and could have used that device.
            if request.submission_seq < admitted_submission_seq
                && request.compatible_devices.contains(&device_ordinal)
                && request.overtakes_remaining > 0
            {
                request.overtakes_remaining -= 1;
            }
        }
        let commit_result = self.restore_exhausted_overtake_protections_locked(state).and_then(|()| {
            super::transaction::publication_deadline_error(publication_deadline, None).map_or(Ok(()), Err)
        });
        if let Err(error) = commit_result {
            let previous_protections = previous.iter().map(|(_, protection)| *protection).collect::<Vec<_>>();
            let error = self.rollback_new_exhausted_overtake_protections(state, &previous_protections, error);
            for (position, request) in state.queued.iter_mut().enumerate() {
                request.overtakes_remaining = previous[position].0;
            }
            return Err(error);
        }
        Ok(())
    }

    /// Returns whether the waiter's required floor is admissible on at least
    /// one compatible device under the normative ledger equation.
    pub(in crate::runtime::query_service::service) fn request_can_receive_required_floor(
        &self,
        _state: &AdmissionState,
        request: &AdmissionRequest,
    ) -> Result<bool> {
        let required_capabilities = super::placement::required_capabilities_for_placement(&request.requirement);
        let Some(resources) = &self.inner.resources else {
            let fallback_device_ordinal = self.default_device_ordinal().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.queue",
                    "admission-only service has no configured device ordinal",
                )
            })?;
            if !self.device_supports_capabilities(fallback_device_ordinal, &required_capabilities) {
                return Ok(false);
            }
            let Some(ledger) = &self.inner.fallback_device_ledger else {
                return Ok(false);
            };
            return ledger
                .candidate_max_new_grant_for_requirement(&request.requirement, request.admission_request_id)
                .map(|max_new_grant| max_new_grant.is_some());
        };
        let placement_request = PlacementRequest::new(0)
            .with_attempt_requirement(request.requirement.clone())
            .with_required_capabilities(required_capabilities);
        Ok(self
            .placement_candidates(resources, &placement_request)?
            .into_iter()
            .any(|candidate| candidate.can_admit_now))
    }

    pub(in crate::runtime::query_service) fn memory_grant_wait_timeout_error(
        &self,
        requirement: &AttemptRequirement,
        queued_queries: usize,
    ) -> QueryError {
        let required_floor = requirement.required_floor_bytes().unwrap_or_default();
        let soft_hint = requirement.soft_working_set_hint_bytes.unwrap_or_default();
        let mut error =
            service_overloaded_error("memory_grant_wait_timeout", queued_queries, self.inner.queue_bounds.max_waiters)
                .with_fact("required_floor_bytes", required_floor)
                .with_fact("soft_working_set_hint_bytes", soft_hint)
                .with_fact("configured_max_active_attempts_per_device", self.max_active_attempt_limit());
        if let Ok(snapshots) = self.device_ledger_snapshots() {
            let live_grants = snapshots
                .iter()
                .map(|snapshot| format!("{}:{}", snapshot.device_ordinal, snapshot.live_grants_bytes))
                .collect::<Vec<_>>()
                .join(",");
            error = error.with_fact("device_live_grants_bytes", live_grants);
            let idle_ceiling = snapshots
                .iter()
                .map(|snapshot| {
                    let reserve = snapshot.configured_floor_bytes.saturating_mul(
                        u64::try_from(snapshot.max_active_attempts.saturating_sub(1)).unwrap_or(u64::MAX),
                    );
                    snapshot.managed_capacity_bytes.saturating_sub(reserve)
                })
                .max()
                .unwrap_or(0);
            error = error.with_fact("guaranteed_min_idle_grant_ceiling_bytes", idle_ceiling);
        }
        error
    }
}

pub(super) fn admission_mutex_poisoned_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Internal,
        "nexus_query_engine.query_service.admission",
        "query service admission mutex was poisoned",
    )
    .with_fact("reason", "admission_mutex_poisoned")
}

pub(in crate::runtime::query_service::service) fn query_service_shutdown_timeout(
    operation: &'static str,
    state: Option<&AdmissionState>,
    deadline_stage: &'static str,
) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for query service shutdown cleanup",
    )
    .with_fact("reason", "query_service_shutdown_timeout")
    .with_fact("deadline_stage", deadline_stage)
    .with_cleanup_deadline_elapsed();
    match state {
        Some(state) => {
            let lifecycle = state.lifecycle.snapshot();
            error
                .with_fact("active_queries", lifecycle.active)
                .with_fact("admitting_queries", lifecycle.provisional)
                .with_fact("cleanup_in_progress", lifecycle.cleanup)
                .with_fact("queued_queries", state.queued.len())
        },
        None => error,
    }
}

/// Test-only rendezvous immediately before an external queue sweep completes
/// its removed requests. This forces the handoff race without timing sleeps.
#[cfg(test)]
#[derive(Debug)]
struct QueueSweepFinishPause {
    entered: std::sync::mpsc::SyncSender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

#[cfg(test)]
impl QueueSweepFinishPause {
    fn wait(&self) {
        self.entered.send(()).expect("queue sweep finish pause receiver remains live");
        self.release
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv()
            .expect("queue sweep finish pause is released");
    }
}

#[cfg(test)]
#[derive(Debug)]
struct QueueSweepFinishPauseRegistration {
    service: std::sync::Weak<QueryServiceInner>,
    pause: Arc<QueueSweepFinishPause>,
}

#[cfg(test)]
static QUEUE_SWEEP_FINISH_PAUSE: std::sync::OnceLock<Mutex<Vec<QueueSweepFinishPauseRegistration>>> =
    std::sync::OnceLock::new();

#[cfg(test)]
fn queue_sweep_finish_pause_registry() -> &'static Mutex<Vec<QueueSweepFinishPauseRegistration>> {
    QUEUE_SWEEP_FINISH_PAUSE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Test-side half of a one-shot queue-sweep pause.
#[cfg(test)]
#[derive(Debug)]
pub(in crate::runtime::query_service) struct QueueSweepFinishPauseHandle {
    entered: Mutex<std::sync::mpsc::Receiver<()>>,
    release: Mutex<Option<std::sync::mpsc::SyncSender<()>>>,
    service: std::sync::Weak<QueryServiceInner>,
    pause: Arc<QueueSweepFinishPause>,
}

#[cfg(test)]
impl QueueSweepFinishPauseHandle {
    pub(in crate::runtime::query_service) fn wait_until_entered(&self, timeout: Duration) {
        self.entered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv_timeout(timeout)
            .expect("queue sweep did not reach the finish pause before the deadline");
    }

    pub(in crate::runtime::query_service) fn release(&self) {
        if let Some(release) = self.release.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            release.send(()).expect("paused queue sweep remains live until release");
        }
    }
}

#[cfg(test)]
impl Drop for QueueSweepFinishPauseHandle {
    fn drop(&mut self) {
        if let Some(release) = self.release.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            let _ = release.send(());
        }
        let mut registration =
            queue_sweep_finish_pause_registry().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(position) = registration.iter().position(|registered| {
            registered.service.ptr_eq(&self.service) && Arc::ptr_eq(&registered.pause, &self.pause)
        }) {
            registration.remove(position);
        }
    }
}

#[cfg(test)]
fn take_queue_sweep_finish_pause(service: &QueryService) -> Option<Arc<QueueSweepFinishPause>> {
    let mut registration =
        queue_sweep_finish_pause_registry().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    registration.retain(|registered| registered.service.upgrade().is_some());
    registration
        .iter()
        .position(|registered| registered.service.upgrade().is_some_and(|target| Arc::ptr_eq(&target, &service.inner)))
        .map(|position| registration.remove(position).pause)
}

#[cfg(test)]
impl QueryService {
    /// Pauses the next sweep from this service after it has removed requests
    /// and recorded their terminal errors, but before completion delivery.
    pub(in crate::runtime::query_service) fn pause_next_queue_sweep_finish_for_test(
        &self,
    ) -> QueueSweepFinishPauseHandle {
        let (entered_sender, entered_receiver) = std::sync::mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
        let pause = Arc::new(QueueSweepFinishPause { entered: entered_sender, release: Mutex::new(release_receiver) });
        let service = Arc::downgrade(&self.inner);
        let mut registration =
            queue_sweep_finish_pause_registry().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        registration.retain(|registered| registered.service.upgrade().is_some());
        assert!(
            !registration.iter().any(|registered| registered.service.ptr_eq(&service)),
            "only one queue sweep finish pause may be armed for a service at a time"
        );
        registration.push(QueueSweepFinishPauseRegistration { service: service.clone(), pause: pause.clone() });
        QueueSweepFinishPauseHandle {
            entered: Mutex::new(entered_receiver),
            release: Mutex::new(Some(release_sender)),
            service,
            pause,
        }
    }
}

/// Removed waiters from one admission sweep. Terminal completion runs after
/// the admission guard is released.
#[derive(Debug, Default)]
pub(in crate::runtime::query_service) struct QueueSweep {
    /// The outcome is captured with the removed request, not recovered from
    /// its waiter-facing error handoff during `finish`.
    pub(in crate::runtime::query_service) expired: Vec<(AdmissionRequest, RuntimeTerminalOutcome)>,
    pub(in crate::runtime::query_service) cancelled: Vec<AdmissionRequest>,
    /// Health rejections also retain their lifecycle projection so waiter
    /// error consumption cannot suppress either terminal side effect.
    pub(in crate::runtime::query_service) terminated:
        Vec<(AdmissionRequest, RuntimeTerminalOutcome, AdmissionLifecycleState)>,
    /// Native faults folded into scheduler ledgers during this sweep. Shared
    /// waiters are notified only after the caller releases its admission guard.
    transitioned_device_ordinals: Vec<usize>,
    #[cfg(test)]
    finish_pause: Option<Arc<QueueSweepFinishPause>>,
}

impl QueueSweep {
    pub(in crate::runtime::query_service) fn is_empty(&self) -> bool {
        self.expired.is_empty()
            && self.cancelled.is_empty()
            && self.terminated.is_empty()
            && self.transitioned_device_ordinals.is_empty()
    }

    /// Terminalizes every removed waiter without holding the admission guard.
    pub(in crate::runtime::query_service) fn finish(mut self) {
        #[cfg(test)]
        if let Some(finish_pause) = &self.finish_pause {
            finish_pause.wait();
        }
        finish_expired_queued_requests(self.expired);
        for request in self.cancelled {
            finish_removed_request_cancelled(Some(request));
        }
        for (request, disposition, rejection) in self.terminated {
            record_admission_lifecycle(&request.observation, request.admission_request_id, rejection, 0, None, None);
            let primary_failure = request.terminal.take_primary_failure();
            request.completion.finish_terminal_disposition(disposition, primary_failure);
        }
        self.transitioned_device_ordinals.sort_unstable();
        self.transitioned_device_ordinals.dedup();
        for device_ordinal in self.transitioned_device_ordinals {
            notify_shared_device_admission_available(device_ordinal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protected_waiter_candidates_are_pinned_to_the_protecting_ledger() {
        assert_eq!(placement_candidate_ordinals(&[0, 1], Some(0)).collect::<Vec<_>>(), vec![0]);
    }
}
