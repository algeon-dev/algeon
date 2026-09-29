//! Shared admission-submission preamble and placed-admission commit.

#[cfg(test)]
use std::sync;
use std::{
    collections::BTreeMap,
    sync::{Arc, MutexGuard, atomic::Ordering},
    time::{Duration, Instant},
};

use query_planning::{
    admission::{AttemptRequirement, ExecutionCapability},
    error::{QueryError, QueryErrorCode, Result},
};

use crate::{
    observability::{
        AdmissionLifecycleState, AdmissionRequestId, AttemptAdmissionWaitFact, AttemptId, RuntimeObservationHandle,
    },
    runtime::{
        ExecutionCancellation,
        backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
        backend_lifecycle::ObservationPermit,
        query_service::service::{
            placement::required_capabilities_for_placement,
            transaction::{
                AdmissionEnvelope, AdmissionTransactionProbe, pause_after_rollback_accounting,
                publication_deadline_error,
            },
        },
    },
};

// `ExecutionCancellation` is intentionally a lightweight shared flag. The
// blocking queue has no caller-owned wake handle, so bounded waits observe it
// at this small fixed interval instead of retaining a stale provisional grant
// until the full queue deadline.
const ADMISSION_CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);

use super::{
    super::{
        admission::{
            AdmissionRequest, AdmissionState, AdmissionTicket, DeviceAdmissionEntries, PendingPreflightReservation,
            PlacedQuery, QueryCompletion, admission_cancellation_or_deadline_error, admission_queue_full_error,
            finish_removed_request_error, service_overloaded_error,
        },
        admission_rejection,
        handle::{QueryHandle, QueryHandleLifetime},
        record_admission_lifecycle,
    },
    QueryService,
    api::QueryAdmissionRequest,
};

mod admit;
mod enqueue;
mod try_admit;

impl QueryService {
    /// Resolves each capability-compatible candidate exactly once before the
    /// request enters queue or capacity ranking.  A device-local owner
    /// preflight failure removes only that candidate; the original structured
    /// owner error is returned when no candidate remains.
    pub(in crate::runtime::query_service::service) fn resolve_device_admission_entries(
        &self,
        request: &QueryAdmissionRequest,
        base_requirement: &AttemptRequirement,
        candidate_ordinals: &[usize],
    ) -> Result<(DeviceAdmissionEntries, Vec<usize>)> {
        #[cfg(test)]
        {
            self.inner.before_pending_preflight_resolution();
            if self.inner.pending_preflight_failure.swap(false, Ordering::AcqRel) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedPlanShape,
                    "query_engine.query_service.admission",
                    "injected device-specific admission preflight failure",
                )
                .with_fact("reason", "injected_pending_preflight_failure"));
            }
        }
        let evidence = request.planned_capability_evidence();
        self.resolve_device_admission_entries_with(candidate_ordinals, |_ordinal| {
            Ok(evidence.cloned().unwrap_or_else(|| base_requirement.clone()))
        })
    }

    fn resolve_device_admission_entries_with<F>(
        &self,
        candidate_ordinals: &[usize],
        mut resolve: F,
    ) -> Result<(DeviceAdmissionEntries, Vec<usize>)>
    where
        F: FnMut(usize) -> Result<AttemptRequirement>,
    {
        let mut entries = DeviceAdmissionEntries::new();
        let mut first_rejection = None;
        for ordinal in candidate_ordinals.iter().copied() {
            match resolve(ordinal) {
                Ok(entry) => {
                    entries.insert(ordinal, entry);
                },
                Err(error) => {
                    let error = error.with_fact_if_absent("device_ordinal", ordinal);
                    first_rejection.get_or_insert(error);
                },
            }
        }
        if entries.is_empty()
            && let Some(error) = first_rejection
        {
            return Err(error.with_fact_if_absent("candidate_count", candidate_ordinals.len()));
        }
        let compatible_devices = entries.keys().copied().collect();
        Ok((entries, compatible_devices))
    }

    #[cfg(test)]
    pub(in crate::runtime::query_service) fn resolve_device_admission_entries_for_test<F>(
        &self,
        candidate_ordinals: &[usize],
        resolve: F,
    ) -> Result<(DeviceAdmissionEntries, Vec<usize>)>
    where
        F: FnMut(usize) -> Result<AttemptRequirement>,
    {
        self.resolve_device_admission_entries_with(candidate_ordinals, resolve)
    }
}

/// All metadata committed with a provisionally placed admission.
///
/// Keeping these values together prevents the three admission paths from
/// accidentally publishing different request metadata.
pub(in crate::runtime::query_service) struct PlacedAdmissionCommit {
    pub(in crate::runtime::query_service) placed_query: PlacedQuery,
    pub(in crate::runtime::query_service) admitted_device_requirements: BTreeMap<usize, AttemptRequirement>,
    pub(in crate::runtime::query_service) submitted_plan_capability_evidence: Option<AttemptRequirement>,
    pub(in crate::runtime::query_service) admitted_compatible_devices: Vec<usize>,
    pub(in crate::runtime::query_service) observation: RuntimeObservationHandle,
    pub(in crate::runtime::query_service) attempt_id: AttemptId,
    pub(in crate::runtime::query_service) completion: QueryCompletion,
    pub(in crate::runtime::query_service) admission_request_id: AdmissionRequestId,
    pub(in crate::runtime::query_service) admission_wait: Duration,
    pub(in crate::runtime::query_service) deadline: Instant,
    pub(in crate::runtime::query_service) execution_cancellation: ExecutionCancellation,
    pub(in crate::runtime::query_service) admission_wait_fact: Option<AttemptAdmissionWaitFact>,
    pub(in crate::runtime::query_service) lifecycle_ticket: AdmissionTicket,
}

struct PreparedSubmission {
    queue_deadline: Option<Instant>,
    requested_query_cap_bytes: Option<u64>,
    requested_at: Instant,
    execution_cancellation: ExecutionCancellation,
    submitted_plan_capability_evidence: Option<AttemptRequirement>,
    request: QueryAdmissionRequest,
    observation: RuntimeObservationHandle,
    attempt_id: AttemptId,
    admission_request_id: AdmissionRequestId,
    completion: QueryCompletion,
    requirement: AttemptRequirement,
    required_capabilities: Vec<ExecutionCapability>,
    candidate_ordinals: Vec<usize>,
}

struct AdmissionPublication {
    observation: RuntimeObservationHandle,
    attempt_id: AttemptId,
    completion: QueryCompletion,
    admitted_device_requirements: BTreeMap<usize, AttemptRequirement>,
    submitted_plan_capability_evidence: Option<AttemptRequirement>,
    admitted_compatible_devices: Vec<usize>,
    admission_request_id: AdmissionRequestId,
    admission_wait: Duration,
    deadline: Instant,
    execution_cancellation: ExecutionCancellation,
    admission_wait_fact: Option<AttemptAdmissionWaitFact>,
}

impl QueryService {
    fn prepare_submission(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<PreparedSubmission> {
        let queue_deadline = request.queue_deadline;
        let requested_query_cap_bytes = request.requested_query_cap_bytes();
        let execution_cancellation = request.execution_cancellation.clone();
        let attempt_requirement = request.resolved_attempt_requirement();
        let submitted_plan_capability_evidence = request.submitted_plan_capability_evidence();
        let mut request = request;
        let mut observation_owner = request.take_runtime_observation();
        let attempt_id = observation_owner.attempt_id.unwrap_or_else(AttemptId::mint);
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let activity_error = observation_owner.activity_error.take();
        let completion = QueryCompletion::new(
            observation_owner.owner.take(),
            observation_owner.attempt_finalizer_slot.take(),
            observation_permit,
            observation_owner.activity.take(),
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        if execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(&execution_cancellation);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                0,
                None,
                None,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                0,
                None,
                None,
            );
            completion.finish_error(&error);
            return Err(error);
        }
        let requirement = match self.resolve_attempt_requirement_for_submission(attempt_requirement) {
            Ok(requirement) => requirement,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                completion.finish_error(&error);
                return Err(error);
            },
        };
        let requested_at = Instant::now();
        let required_capabilities = required_capabilities_for_placement(&requirement);
        let candidate_ordinals = self.compatible_devices_for_capabilities(&required_capabilities)?;
        Ok(PreparedSubmission {
            queue_deadline,
            requested_query_cap_bytes,
            requested_at,
            execution_cancellation,
            submitted_plan_capability_evidence,
            request,
            observation: observation_owner.handle,
            attempt_id,
            admission_request_id,
            completion,
            requirement,
            required_capabilities,
            candidate_ordinals,
        })
    }

    fn reserve_pending_preflight_for_request(
        &self,
        observation: &RuntimeObservationHandle,
        admission_request_id: AdmissionRequestId,
        execution_cancellation: &ExecutionCancellation,
        deadline: Instant,
    ) -> Result<PendingPreflightReservation> {
        let mut state = self.lock_admission()?;
        let sweep = self.sweep_expired_queued_requests_locked(&mut state);
        if !sweep.is_empty() {
            // Queue-sweep completion may reenter service shutdown, so release the admission lock first.
            drop(state);
            sweep.finish();
            state = self.lock_admission()?;
        }
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        if execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(execution_cancellation);
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                state.queued.len(),
                None,
                None,
            );
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        if Instant::now() >= deadline {
            let error =
                service_overloaded_error("queue_wait_elapsed", state.queued.len(), self.inner.queue_bounds.max_waiters);
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::TimedOut,
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        let Some(pending) = self.reserve_pending_preflight_locked(&mut state) else {
            let error = admission_queue_full_error(
                state.queued.len(),
                self.inner.queue_bounds.max_waiters,
                state.pending_preflight.len(),
            );
            record_admission_lifecycle(
                observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        };
        Ok(pending)
    }

    fn convert_pending_preflight_to_waiter(
        &self,
        request: AdmissionRequest,
        pending: PendingPreflightReservation,
    ) -> Result<MutexGuard<'_, AdmissionState>> {
        let mut state = self.lock_admission()?;
        let error = if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                &request.observation,
                request.admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            Some(error)
        } else if request.execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(&request.execution_cancellation);
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
            Some(error)
        } else if Instant::now() >= request.deadline {
            let error =
                service_overloaded_error("queue_wait_elapsed", state.queued.len(), self.inner.queue_bounds.max_waiters);
            record_admission_lifecycle(
                &request.observation,
                request.admission_request_id,
                AdmissionLifecycleState::TimedOut,
                state.queued.len(),
                None,
                None,
            );
            Some(error)
        } else {
            None
        };
        if let Some(error) = error {
            // Completion may reenter shutdown, so release the lock and pending reservation first.
            drop(state);
            pending.release();
            finish_removed_request_error(Some(request), &error);
            return Err(error);
        }
        let observation = request.observation.clone();
        let admission_request_id = request.admission_request_id;
        state.enqueue_request(request);
        pending.convert_locked(&mut state);
        self.inner.notify_admission_available();
        record_admission_lifecycle(
            &observation,
            admission_request_id,
            AdmissionLifecycleState::Enqueued,
            state.queued.len(),
            None,
            None,
        );
        Ok(state)
    }

    /// Shared admission-commit tail. The caller has selected and placed the
    /// request, removed it from the queue, and provisionally
    /// counted it as admitting; the admission mutex is not held. Builds the
    /// admission envelope; lane, root stream, allocation domain, root
    /// member, worker scope; while the mutex is released, then publishes
    /// admission. Any failure restores the provisional count exactly, releases
    /// every envelope component in reverse order outside the mutex, and
    /// finishes the request with the primary error.
    pub(in crate::runtime::query_service) fn commit_placed_admission(
        &self,
        commit: PlacedAdmissionCommit,
    ) -> Result<QueryHandle> {
        let PlacedAdmissionCommit {
            placed_query,
            admitted_device_requirements,
            submitted_plan_capability_evidence,
            admitted_compatible_devices,
            observation,
            attempt_id,
            completion,
            admission_request_id,
            admission_wait,
            deadline,
            execution_cancellation,
            admission_wait_fact,
            lifecycle_ticket,
        } = commit;
        #[cfg(test)]
        let probe = self.inner.admission_transaction_probe.lock().unwrap_or_else(sync::PoisonError::into_inner).clone();
        // Production installs no probe; the admission transaction hooks are
        // inlined no-ops over a constant `None`.
        #[cfg(not(test))]
        let probe: Option<Arc<AdmissionTransactionProbe>> = None;
        #[cfg(test)]
        if self.inner.direct_placement_resolution_failure.swap(false, Ordering::AcqRel)
            && let Some(pause) =
                self.inner.direct_provisional_cleanup_pause.lock().unwrap_or_else(sync::PoisonError::into_inner).take()
        {
            let mut lifecycle_ticket = lifecycle_ticket;
            let PlacedQuery { device_ordinal: _, memory_grant } = placed_query;
            let state = self.inner.admission.lock().unwrap_or_else(sync::PoisonError::into_inner);
            lifecycle_ticket.transition_to_cleanup().expect("injected resolution failure accounts provisional cleanup");
            self.inner.notify_admission_available();
            drop(state);
            pause.wait();
            drop(memory_grant);
            drop(lifecycle_ticket);
            let error = QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.query_service.placement",
                "device-specific admission resolution was unavailable",
            )
            .with_fact("reason", "missing_device_specific_admission_resolution");
            completion.finish_error(&error);
            return Err(error);
        }
        let envelope = AdmissionEnvelope::acquire(
            &self.inner,
            placed_query,
            lifecycle_ticket,
            &execution_cancellation,
            deadline,
            &observation,
            probe.as_deref(),
        );
        let state = self.inner.admission.lock().unwrap_or_else(|error| {
            eprintln!("[query-engine] warning: recovering poisoned admission state during admission commit");
            error.into_inner()
        });
        let envelope = match envelope {
            Ok(envelope) => envelope,
            Err(primary) => {
                return Err(self.finish_rejected_envelope(
                    state,
                    primary,
                    &observation,
                    admission_request_id,
                    completion,
                ));
            },
        };
        self.publish_admission_envelope(
            state,
            envelope,
            AdmissionPublication {
                observation,
                attempt_id,
                completion,
                admitted_device_requirements,
                submitted_plan_capability_evidence,
                admitted_compatible_devices,
                admission_request_id,
                admission_wait,
                deadline,
                execution_cancellation,
                admission_wait_fact,
            },
            &probe,
        )
    }

    fn finish_rejected_envelope(
        &self,
        state: MutexGuard<'_, AdmissionState>,
        primary: QueryError,
        observation: &RuntimeObservationHandle,
        admission_request_id: AdmissionRequestId,
        completion: QueryCompletion,
    ) -> QueryError {
        if primary.code() == QueryErrorCode::Cancelled {
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                state.queued.len(),
                None,
                None,
            );
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                state.queued.len(),
                None,
                None,
            );
        } else if primary.fact_value("reason") == Some("queue_wait_elapsed") {
            record_admission_lifecycle(
                observation,
                admission_request_id,
                AdmissionLifecycleState::TimedOut,
                state.queued.len(),
                None,
                None,
            );
        } else {
            record_admission_lifecycle(
                observation,
                admission_request_id,
                admission_rejection(&primary),
                state.queued.len(),
                None,
                None,
            );
        }
        self.inner.notify_admission_available();
        drop(state);
        if primary.fact_value("reason") == Some("queue_wait_elapsed") {
            completion.finish_timeout(&primary);
        } else {
            completion.finish_error(&primary);
        }
        primary
    }

    fn publish_admission_envelope(
        &self,
        mut state: MutexGuard<'_, AdmissionState>,
        mut envelope: AdmissionEnvelope,
        publication: AdmissionPublication,
        probe: &Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<QueryHandle> {
        let AdmissionPublication {
            observation,
            attempt_id,
            completion,
            admitted_device_requirements,
            submitted_plan_capability_evidence,
            admitted_compatible_devices,
            admission_request_id,
            admission_wait,
            deadline,
            execution_cancellation,
            admission_wait_fact,
        } = publication;
        let admitted_capability_evidence = submitted_plan_capability_evidence.map(Arc::new);
        envelope.register_allocator_domain_attempt(attempt_id);
        if let Err(error) = execution_cancellation.check("query_engine.query_service.admission_transaction.publish") {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                state.queued.len(),
                None,
                None,
            );
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                AdmissionLifecycleState::Cancelled,
                state.queued.len(),
                None,
                None,
            );
            self.inner.notify_admission_available();
            drop(state);
            pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        if let Some(error) = self.admission_terminal_rejection(&state) {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            self.inner.notify_admission_available();
            drop(state);
            // Reverse-order component release happens outside the admission
            // mutex: the grant drop invokes the ledger release hook.
            pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        if let Some(error) = publication_deadline_error(deadline, probe.as_deref()) {
            envelope.preaccount_rollback_cleanup(&mut state);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                AdmissionLifecycleState::TimedOut,
                state.queued.len(),
                None,
                None,
            );
            self.inner.notify_admission_available();
            drop(state);
            pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_timeout(&error);
            return Err(error);
        }
        if let Err(error) = envelope.activate_lifecycle_ticket() {
            envelope.preaccount_rollback_cleanup(&mut state);
            self.inner.notify_admission_available();
            drop(state);
            pause_after_rollback_accounting(probe.as_deref());
            drop(envelope);
            self.inner.notify_admission_available();
            completion.finish_error(&error);
            return Err(error);
        }
        let completion = match completion.into_admitted(&mut envelope) {
            Ok(completion) => completion,
            Err(error) => {
                // A duplicate binding cannot manufacture a second post-cleanup
                // finalizer. The provisional envelope rolls back and the
                // admission is rejected; missing bookkeeping never closes the
                // fleet, which only CUDA fault evidence can do.
                envelope.preaccount_rollback_cleanup(&mut state);
                record_admission_lifecycle(
                    &observation,
                    admission_request_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                self.inner.notify_admission_available();
                drop(state);
                pause_after_rollback_accounting(probe.as_deref());
                drop(envelope);
                self.inner.notify_admission_available();
                return Err(error);
            },
        };
        let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
        let device_ordinal = envelope.device_ordinal();
        let device_budget_bytes = envelope.memory_grant().budget_bytes();
        record_admission_lifecycle(
            &observation,
            admission_request_id,
            AdmissionLifecycleState::Admitted,
            state.queued.len(),
            Some(device_ordinal),
            Some(device_budget_bytes),
        );
        // Release admission lock before constructing and returning the handle.
        drop(state);
        Ok(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&self.inner),
                admission_request_id,
                observation,
                attempt_id,
                completion,
                envelope,
                admitted_capability_evidence,
                admitted_device_requirements,
                admitted_compatible_devices,
            ),
            config: self.inner.config.clone(),
            queue_bounds: self.inner.queue_bounds.clone(),
            admission_request_id,
            admission_sequence,
            admission_wait,
            admission_wait_fact,
            device_budget_bytes,
            device_ordinal,
        })
    }
}
