//! Admission queue selection, sweep, and lock helpers.
//!
//! Queue order is strictly oldest-first by submission sequence. The first
//! runnable waiter wins on each compatible device; a blocked waiter does not
//! idle unrelated devices needed only by later work.

use std::{
    collections::{BTreeSet, VecDeque},
    sync::{Arc, MutexGuard, atomic::Ordering},
    time::Instant,
};

use query_planning::{
    admission::{AttemptRequirement, ExecutionCapability, capability_labels},
    error::{QueryError, QueryErrorCode, Result},
};

use super::{
    super::{
        admission::{
            AdmissionRequest, AdmissionState, DeviceAdmissionEntries, PendingPreflightReservation,
            admission_cancellation_or_deadline_error, finish_expired_queued_requests, finish_removed_request_cancelled,
            finish_removed_request_error, remove_queued_request, service_closed_error, service_overloaded_error,
        },
        admission_rejection, record_admission_lifecycle,
    },
    AdmissionController,
};
use crate::{
    observability::{AdmissionLifecycleState, AdmissionRequestId, AttemptFaultIsolation, RuntimeTerminalOutcome},
    runtime::{DeviceLedger, deadline::mutex_lock_recovering_until, gpu::DeviceAdmissionHealth},
};

impl AdmissionController {
    pub(in crate::runtime::query_service) fn reserve_pending_preflight_locked(
        self: &Arc<Self>,
        state: &mut AdmissionState,
    ) -> Option<PendingPreflightReservation> {
        if state.queued.len() + state.pending_preflight.len() >= self.queue_bounds.max_waiters {
            return None;
        }
        let submission_seq = self.next_submission_seq.fetch_add(1, Ordering::Relaxed);
        let inserted = state.pending_preflight.insert(submission_seq);
        debug_assert!(inserted, "pending preflight submission sequences must be unique");
        Some(PendingPreflightReservation::new(Arc::downgrade(self), submission_seq))
    }

    pub(in crate::runtime::query_service) fn pending_preflight_min(&self, state: &AdmissionState) -> Option<u64> {
        state.pending_preflight.first().copied()
    }

    pub(in crate::runtime::query_service) fn remove_queued_request_without_admission_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        remove_queued_request(state, admission_request_id)
    }

    /// Computes the immutable capability-compatible device set at submission.
    ///
    /// This intentionally precedes queue insertion: a requirement with no
    /// capable installed device is a permanent local rejection, never a
    /// capacity waiter. Health and ledger capacity are evaluated afterwards
    /// while selecting among this frozen set.
    pub(in crate::runtime::query_service) fn compatible_devices_for_capabilities(
        &self,
        required_capabilities: &[ExecutionCapability],
    ) -> Result<Vec<usize>> {
        let devices = self
            .devices
            .ledgers()
            .map(|ledger| ledger.device_ordinal())
            .filter(|ordinal| self.devices.supports_capabilities(*ordinal, required_capabilities))
            .collect();
        Ok(devices)
    }

    /// Resolves the request's immutable requirement without allowing a missing
    /// capability requirement to pass through an exact device profile as an empty set.
    pub(in crate::runtime::query_service) fn resolve_attempt_requirement_for_submission(
        &self,
        attempt_requirement: Option<AttemptRequirement>,
    ) -> Result<AttemptRequirement> {
        if let Some(requirement) = attempt_requirement {
            return Ok(requirement);
        }
        let restricted_device_count = self
            .devices
            .ledgers()
            .filter(|ledger| ledger.profile().device_capabilities.requires_explicit_attempt_requirement())
            .count();
        if restricted_device_count == 0 {
            return Ok(AttemptRequirement::empty());
        }
        Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.query_service.admission",
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
        required_capabilities: &[ExecutionCapability],
    ) -> bool {
        self.devices.supports_capabilities(device_ordinal, required_capabilities)
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
                DeviceAdmissionHealth::Open => {
                    return Ok(None);
                },
                DeviceAdmissionHealth::Closed { reason, isolation, generation } => {
                    all_closed_for_unknown_isolation &= isolation == AttemptFaultIsolation::Unknown;
                    first_closed_ledger.get_or_insert((*ordinal, reason, generation));
                },
            }
        }

        if let Some((device_ordinal, detail, generation)) = first_closed_ledger {
            let mut error = QueryError::new(
                QueryErrorCode::CudfPersistentFailure,
                "query_engine.query_service.admission",
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
                error = error.with_fact("detail", detail);
            }
            return Ok(Some(error));
        }
        Ok(None)
    }

    pub(in crate::runtime::query_service) fn no_capable_device_error(
        &self,
        required_capabilities: &[ExecutionCapability],
    ) -> QueryError {
        QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.query_service.admission",
            "no installed device supports every required native capability",
        )
        .with_fact("reason", "required_device_capability_unavailable")
        .with_fact("required_capabilities", capability_labels(required_capabilities))
        .with_fact("configured_device_count", self.devices.len())
        .with_fact("compatible_device_count", 0)
    }

    pub(in crate::runtime::query_service) fn reject_impossible_requirements(
        &self,
        device_entries: &DeviceAdmissionEntries,
        required_capabilities: &[ExecutionCapability],
        requested_query_cap_bytes: Option<u64>,
    ) -> Result<()> {
        if device_entries.is_empty() && !required_capabilities.is_empty() {
            return Err(self.no_capable_device_error(required_capabilities));
        }
        let mut can_ever_fit = false;
        let mut maximum_resident_capacity = 0u64;
        let mut open_device_count = 0usize;
        for ordinal in device_entries.keys() {
            let ledger = self.ledger_for_ordinal(*ordinal)?;
            let snapshot = ledger.snapshot()?;
            // A closed device can never run this attempt, so its capacity
            // cannot keep an impossible cap waiting. Direct placement already
            // skips closed devices; the waiter revalidation must agree, or a
            // single closed device makes every cap look satisfiable until the
            // waiter times out.
            if !matches!(snapshot.health, DeviceAdmissionHealth::Open) {
                continue;
            }
            open_device_count += 1;
            maximum_resident_capacity = maximum_resident_capacity
                .max(snapshot.query_share_bytes.saturating_sub(snapshot.frozen_resident_bytes));
            can_ever_fit |= snapshot.can_ever_fit_query_cap(requested_query_cap_bytes);
        }
        // A fully closed fleet is a health failure, not an impossible cap. Let
        // the canonical closed-fleet error own that case.
        if open_device_count == 0 && !device_entries.is_empty() {
            return Ok(());
        }
        if device_entries.is_empty() || !can_ever_fit {
            return Err(QueryError::new(
                QueryErrorCode::QueryBudgetUnavailable,
                "query_engine.query_service.admission",
                "no compatible device can satisfy the requested query cap after resident ownership",
            )
            .with_fact("reason", "query_cap_exceeds_resident_capacity")
            .with_fact("requested_query_cap_bytes", requested_query_cap_bytes.unwrap_or_default())
            .with_fact("max_compatible_resident_capacity_bytes", maximum_resident_capacity));
        }
        Ok(())
    }

    pub(in crate::runtime::query_service) fn ledger_for_ordinal(&self, ordinal: usize) -> Result<Arc<DeviceLedger>> {
        self.devices.ledger(ordinal).map(Arc::clone)
    }

    /// The device-state rejection of `request`, if a ledger transition since
    /// its last check made it impossible. A request is checked when it enters
    /// the queue and again after each transition that can make it impossible.
    pub(in crate::runtime::query_service) fn revalidate_queued_request_device_state(
        &self,
        request: &mut AdmissionRequest,
    ) -> Option<QueryError> {
        let generation = self.queue_revalidation_generation.load(Ordering::Acquire);
        if request.validated_generation == Some(generation) {
            return None;
        }
        match self.all_compatible_devices_closed_error(&request.compatible_devices) {
            Ok(Some(error)) => Some(error),
            Ok(None) => {
                match self.reject_impossible_requirements(
                    &request.device_entries,
                    &[],
                    request.requested_query_cap_bytes,
                ) {
                    Ok(()) => {
                        request.validated_generation = Some(generation);
                        None
                    },
                    Err(error) if error.code() == QueryErrorCode::QueryBudgetUnavailable => Some(error),
                    Err(error) => {
                        eprintln!(
                            "[query-engine] warning: failed to revalidate queued resident capacity; retaining waiter: {error}"
                        );
                        None
                    },
                }
            },
            Err(error) => {
                eprintln!("[query-engine] warning: failed to evaluate queued-device health; retaining waiter: {error}");
                None
            },
        }
    }

    /// Waiters that are queue members: neither cancelled nor past their
    /// deadline.
    pub(in crate::runtime::query_service) fn queue_member_count_locked(&self, state: &AdmissionState) -> usize {
        let now = Instant::now();
        state.queued.iter().filter(|request| request.is_queue_member(now)).count()
    }

    pub(in crate::runtime::query_service) fn abandon_queued_request_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        let removed = self.remove_queued_request_without_admission_locked(state, admission_request_id);
        if removed.is_some() {
            self.notify_admission_available();
        }
        removed
    }

    pub(in crate::runtime::query_service) fn lock_admission(&self) -> Result<MutexGuard<'_, AdmissionState>> {
        self.state.lock().map_err(|_error| admission_mutex_poisoned_error())
    }

    /// Removes and terminalizes one queued request after recovering a poisoned
    /// admission guard. The terminal callback runs only after the guard is
    /// released, because it may synchronously reenter backend shutdown.
    pub(in crate::runtime::query_service) fn recover_queued_request_after_admission_poison(
        &self,
        mut state: MutexGuard<'_, AdmissionState>,
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
        !self.admission_open.load(Ordering::Acquire) || state.closed
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

    pub(in crate::runtime::query_service) fn lock_admission_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, AdmissionState>> {
        mutex_lock_recovering_until(&self.state, deadline)
            .ok_or_else(|| query_service_shutdown_timeout(operation, None, "admission_mutex"))
    }

    pub(in crate::runtime::query_service) fn sweep_expired_queued_requests_locked(
        &self,
        state: &mut AdmissionState,
    ) -> QueueSweep {
        if let Err(error) = self.reconcile_native_admission_health() {
            eprintln!("[query-engine] warning: native admission-health reconciliation failed: {error}");
        }
        // Terminal intent has a strict precedence. A caller that cancelled or
        // expired while queued must be removed before a concurrent health
        // transition must not replace that terminal result with a
        // persistent-device failure.
        let now = Instant::now();
        let mut expired = Vec::new();
        let mut cancelled = Vec::new();
        let mut queued = state.take_queued_requests();
        let mut retained = VecDeque::with_capacity(queued.len());
        while let Some(request) = queued.pop_front() {
            if request.execution_cancellation.is_cancelled() {
                let _ =
                    request.terminal.record(admission_cancellation_or_deadline_error(&request.execution_cancellation));
                cancelled.push(request);
            } else if request.deadline.is_some_and(|deadline| deadline <= now) {
                expired.push(request);
            } else {
                retained.push_back(request);
            }
        }
        // The queue depth an expired waiter reports is the queue's members
        // when it ended: the waiters neither cancelled nor expired.
        let members = retained.len();
        let expired = expired
            .into_iter()
            .map(|request| {
                let disposition = request.terminal.record(service_overloaded_error(
                    "queue_wait_elapsed",
                    members,
                    self.queue_bounds.max_waiters,
                ));
                (request, disposition)
            })
            .collect::<Vec<_>>();
        state.replace_queued_requests(retained);

        // Device health and resident capacity change only through ledger
        // transitions. A request is checked once when it enters the queue and
        // again after a transition that can make it impossible; the
        // reconciliation above may itself have closed a ledger.
        let mut terminated = Vec::new();
        let mut queued = state.take_queued_requests();
        let mut retained = VecDeque::with_capacity(queued.len());
        while let Some(mut request) = queued.pop_front() {
            match self.revalidate_queued_request_device_state(&mut request) {
                Some(error) => terminated.push((request, error)),
                None => retained.push_back(request),
            }
        }
        state.replace_queued_requests(retained);
        if !expired.is_empty() {
            self.notify_admission_available();
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
            self.notify_admission_available();
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
            self.notify_admission_available();
        }
        let mut terminated_requests = Vec::with_capacity(terminated.len());
        for (request, error) in terminated {
            let rejection = admission_rejection(&error);
            let disposition = request.terminal.record(error);
            terminated_requests.push((request, disposition, rejection));
        }
        QueueSweep { expired, cancelled, terminated: terminated_requests }
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
        let pending_barrier = self.pending_preflight_min(state);
        let now = Instant::now();
        let mut devices_claimed_by_older_waiters = BTreeSet::new();
        for (position, request) in state.queued.iter().enumerate() {
            if pending_barrier.is_some_and(|pending| request.submission_seq >= pending) {
                break;
            }
            // A cancelled or expired waiter is no longer a queue member: it
            // claims no device ahead of younger waiters.
            if !request.is_queue_member(now) {
                continue;
            }
            if let Some(best) = self.best_eligible_device_for_excluding(request, &devices_claimed_by_older_waiters)? {
                return Ok(Some((position, best)));
            }
            devices_claimed_by_older_waiters.extend(request.compatible_devices.iter().copied());
        }
        Ok(None)
    }

    fn best_eligible_device_for_excluding(
        &self,
        request: &AdmissionRequest,
        excluded_devices: &BTreeSet<usize>,
    ) -> Result<Option<usize>> {
        let candidates = self.placement_candidates_for_entries(
            &request.device_entries,
            request.compatible_devices.iter().copied().filter(|ordinal| !excluded_devices.contains(ordinal)),
            request.requested_query_cap_bytes,
        )?;
        Ok(candidates
            .into_iter()
            .find(|candidate| candidate.can_admit_now)
            .map(|candidate| candidate.candidate.ordinal))
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
        let now = Instant::now();
        let Some((position, first)) = state.queued.iter().enumerate().find(|(_, request)| request.is_queue_member(now))
        else {
            return Ok(None);
        };
        if self.pending_preflight_min(state).is_some_and(|pending| first.submission_seq >= pending) {
            return Ok(None);
        }
        if first.admission_request_id != admission_request_id {
            return Ok(None);
        }
        // Same ranking as normal queue and direct placement. Unlike ordinary
        // selection the fallback deliberately drives the best healthy
        // candidate even if it is saturated, so the resulting wait retains the
        // memory-grant timeout taxonomy.
        let candidates = self.placement_candidates_for_entries(
            &first.device_entries,
            first.compatible_devices.iter().copied(),
            first.requested_query_cap_bytes,
        )?;
        Ok(candidates.into_iter().next().map(|candidate| (position, candidate.candidate.ordinal)))
    }

    /// Reports the device the waiter was placed on but could not obtain a
    /// grant from.
    ///
    /// The facts are scoped to that one selected device: a fleet-wide
    /// projection would also report devices this waiter was never eligible
    /// for, and `admissible_query_cap_bytes` does not consult device health,
    /// so a closed device would advertise spare capacity it can never issue.
    pub(in crate::runtime::query_service) fn memory_grant_wait_timeout_error(
        &self,
        device_ordinal: usize,
        requested_query_cap_bytes: Option<u64>,
        queued_queries: usize,
    ) -> QueryError {
        let mut error =
            service_overloaded_error("memory_grant_wait_timeout", queued_queries, self.queue_bounds.max_waiters)
                .with_fact("requested_query_cap_bytes", requested_query_cap_bytes.unwrap_or_default());
        if let Ok(snapshot) = self.ledger_for_ordinal(device_ordinal).and_then(|ledger| ledger.snapshot()) {
            let ordinal = snapshot.device_ordinal;
            let cap = snapshot.admissible_query_cap_bytes(requested_query_cap_bytes).unwrap_or_default();
            error = error
                .with_fact("device_frozen_resident_bytes", format!("{ordinal}:{}", snapshot.frozen_resident_bytes))
                .with_fact("device_live_grants_bytes", format!("{ordinal}:{}", snapshot.live_grants_bytes))
                .with_fact("device_admissible_query_cap_bytes", format!("{ordinal}:{cap}"));
        }
        error
    }
}

pub(in crate::runtime::query_service) fn admission_mutex_poisoned_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Internal,
        "query_engine.query_service.admission",
        "query service admission mutex was poisoned",
    )
    .with_fact("reason", "admission_mutex_poisoned")
}

pub(in crate::runtime::query_service) fn query_service_shutdown_timeout(
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
                .with_fact("pending_preflight", state.pending_preflight.len())
        },
        None => error,
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
}

impl QueueSweep {
    pub(in crate::runtime::query_service) fn is_empty(&self) -> bool {
        self.expired.is_empty() && self.cancelled.is_empty() && self.terminated.is_empty()
    }

    /// Terminalizes every removed waiter without holding the admission guard.
    pub(in crate::runtime::query_service) fn finish(self) {
        finish_expired_queued_requests(self.expired);
        for request in self.cancelled {
            finish_removed_request_cancelled(Some(request));
        }
        for (request, disposition, rejection) in self.terminated {
            record_admission_lifecycle(&request.observation, request.admission_request_id, rejection, 0, None, None);
            let primary_failure = request.terminal.take_primary_failure();
            request.completion.finish_terminal_disposition(disposition, primary_failure);
        }
    }
}
