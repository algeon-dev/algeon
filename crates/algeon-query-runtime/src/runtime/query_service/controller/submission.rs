//! Admission submission: the attempt claim, requirement resolution, and the
//! pending-preflight window that turns a request into a queued waiter.

use std::{
    sync::{Arc, MutexGuard, atomic::Ordering},
    time::Instant,
};

use query_planning::{
    admission::{AttemptCapabilityEvidence, ExecutionCapability},
    error::{QueryError, QueryErrorCode, Result},
};

use super::{
    super::{
        admission::{
            AdmissionRequest, AdmissionState, AdmissionWaiter, DeviceAdmissionEntries, PendingPreflightReservation,
            QueryCompletion, QueuedAdmissionTerminal, admission_cancellation_or_deadline_error,
            admission_queue_full_error, finish_removed_request_error, service_overloaded_error,
        },
        admission_rejection, record_admission_lifecycle,
        service::{AdmissionObservation, QueryAdmissionRequest},
    },
    AdmissionController, required_capabilities_for_placement,
};
use crate::{
    attempt::AttemptLifecycleError,
    observability::{AdmissionLifecycleState, AdmissionRequestId, AttemptId, RuntimeObservationHandle},
    runtime::{
        ExecutionCancellation,
        backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
        backend_lifecycle::ObservationPermit,
    },
};

#[derive(Debug)]
struct PreparedSubmission {
    queue_deadline: Option<Instant>,
    requested_query_cap_bytes: Option<u64>,
    requested_at: Instant,
    execution_cancellation: ExecutionCancellation,
    submitted_plan_capability_evidence: Option<AttemptCapabilityEvidence>,
    observation: RuntimeObservationHandle,
    attempt_id: AttemptId,
    admission_request_id: AdmissionRequestId,
    completion: QueryCompletion,
    requirement: AttemptCapabilityEvidence,
    required_capabilities: Vec<ExecutionCapability>,
    candidate_ordinals: Vec<usize>,
}

/// A prepared submission holding a pending-preflight slot. Until
/// [`AdmissionController::finish_submission`] queues or rejects it, the slot
/// counts against the queue bound and keeps younger work from overtaking it.
#[derive(Debug)]
pub(in crate::runtime::query_service) struct PendingSubmission {
    prepared: PreparedSubmission,
    deadline: Option<Instant>,
    pending: PendingPreflightReservation,
}

impl AdmissionController {
    /// Submits one request: claims the attempt's admission, resolves its
    /// device requirement and compatible devices, and enters the queue
    /// through the pending-preflight window. A request rejected before
    /// queueing finishes its completion with the error.
    pub(in crate::runtime::query_service) fn enqueue(
        self: &Arc<Self>,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<AdmissionWaiter> {
        let submission =
            self.begin_submission(request, observation_permit, gpu_resource_permit, completion_dispatcher)?;
        self.finish_submission(submission)
    }

    /// Claims the attempt's admission, resolves its requirement and
    /// capability-compatible devices, and reserves a pending-preflight slot.
    pub(in crate::runtime::query_service) fn begin_submission(
        self: &Arc<Self>,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<PendingSubmission> {
        let prepared =
            self.prepare_submission(request, observation_permit, gpu_resource_permit, completion_dispatcher)?;
        let deadline = self.queue_bounds.wait_deadline(prepared.requested_at, prepared.queue_deadline);
        match self.reserve_pending_preflight_for_request(
            &prepared.observation,
            prepared.admission_request_id,
            &prepared.execution_cancellation,
            deadline,
        ) {
            Ok(pending) => Ok(PendingSubmission { prepared, deadline, pending }),
            Err(error) => {
                prepared.completion.finish_error(&error);
                Err(error)
            },
        }
    }

    /// Resolves the device entries of a pending submission and converts its
    /// slot into a queued waiter, unless admission closed, the request was
    /// cancelled, or its queue deadline passed meanwhile.
    pub(in crate::runtime::query_service) fn finish_submission(
        self: &Arc<Self>,
        submission: PendingSubmission,
    ) -> Result<AdmissionWaiter> {
        let entry = submission
            .prepared
            .submitted_plan_capability_evidence
            .clone()
            .unwrap_or_else(|| submission.prepared.requirement.clone());
        self.finish_submission_with(submission, |_ordinal| Ok(entry.clone()))
    }

    /// [`Self::finish_submission`] with the per-device entry resolution
    /// supplied by the caller.
    pub(in crate::runtime::query_service) fn finish_submission_with<F>(
        self: &Arc<Self>,
        submission: PendingSubmission,
        resolve: F,
    ) -> Result<AdmissionWaiter>
    where
        F: FnMut(usize) -> Result<AttemptCapabilityEvidence>,
    {
        let PendingSubmission { prepared, deadline, pending } = submission;
        let PreparedSubmission {
            queue_deadline: _,
            requested_query_cap_bytes,
            requested_at,
            execution_cancellation,
            submitted_plan_capability_evidence,
            observation,
            attempt_id,
            admission_request_id,
            completion,
            requirement: _,
            required_capabilities,
            candidate_ordinals,
        } = prepared;
        let submission_seq = pending.submission_seq();
        let (device_entries, compatible_devices) =
            match self.resolve_device_admission_entries_with(&candidate_ordinals, resolve) {
                Ok(entries) => entries,
                Err(error) => {
                    record_admission_lifecycle(
                        &observation,
                        admission_request_id,
                        admission_rejection(&error),
                        0,
                        None,
                        None,
                    );
                    pending.release();
                    completion.finish_error(&error);
                    return Err(error);
                },
            };
        if let Err(error) =
            self.reject_impossible_requirements(&device_entries, &required_capabilities, requested_query_cap_bytes)
        {
            record_admission_lifecycle(&observation, admission_request_id, admission_rejection(&error), 0, None, None);
            pending.release();
            completion.finish_error(&error);
            return Err(error);
        }

        let request = AdmissionRequest {
            admission_request_id,
            observation: observation.clone(),
            attempt_id,
            completion,
            submitted_plan_capability_evidence,
            device_entries,
            submission_seq,
            compatible_devices,
            requested_query_cap_bytes,
            deadline,
            execution_cancellation: execution_cancellation.clone(),
            terminal: QueuedAdmissionTerminal::default(),
            validated_generation: None,
        };
        let terminal = request.terminal.clone();
        {
            let _state = self.convert_pending_preflight_to_waiter(request, pending)?;
        }
        Ok(AdmissionWaiter {
            controller: Arc::clone(self),
            admission_request_id,
            observation,
            requested_at,
            deadline,
            execution_cancellation,
            terminal,
            queued: true,
        })
    }

    /// Resolves each capability-compatible candidate exactly once before the
    /// request enters queue or capacity ranking. A device-local resolution
    /// failure removes only that candidate; the first structured error is
    /// returned when no candidate remains.
    pub(in crate::runtime::query_service) fn resolve_device_admission_entries_with<F>(
        &self,
        candidate_ordinals: &[usize],
        mut resolve: F,
    ) -> Result<(DeviceAdmissionEntries, Vec<usize>)>
    where
        F: FnMut(usize) -> Result<AttemptCapabilityEvidence>,
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

    fn prepare_submission(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<PreparedSubmission> {
        // The claim runs before any queue, ledger, stream, or lifecycle
        // change; a rejected claim leaves the attempt unchanged.
        let attempt_admission = request.claim_attempt_admission().map_err(attempt_admission_claim_error)?;
        let queue_deadline = request.queue_deadline;
        let requested_query_cap_bytes = request.requested_query_cap_bytes();
        let execution_cancellation = request.execution_cancellation.clone();
        let attempt_requirement = request.resolved_attempt_requirement();
        let submitted_plan_capability_evidence = request.submitted_plan_capability_evidence();
        let AdmissionObservation { handle: observation, activity, activity_error, attempt_id } =
            request.into_admission_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        let completion = QueryCompletion::new(
            attempt_admission,
            observation_permit,
            activity,
            completion_dispatcher,
            gpu_resource_permit,
        );
        if let Some(error) = activity_error {
            record_admission_lifecycle(&observation, admission_request_id, admission_rejection(&error), 0, None, None);
            completion.finish_error(&error);
            return Err(error);
        }
        if execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(&execution_cancellation);
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                0,
                None,
                None,
            );
            record_admission_lifecycle(
                &observation,
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
                    &observation,
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
            observation,
            attempt_id,
            admission_request_id,
            completion,
            requirement,
            required_capabilities,
            candidate_ordinals,
        })
    }

    fn reserve_pending_preflight_for_request(
        self: &Arc<Self>,
        observation: &RuntimeObservationHandle,
        admission_request_id: AdmissionRequestId,
        execution_cancellation: &ExecutionCancellation,
        deadline: Option<Instant>,
    ) -> Result<PendingPreflightReservation> {
        let mut state = self.lock_admission()?;
        let sweep = self.sweep_expired_queued_requests_locked(&mut state);
        if !sweep.is_empty() {
            // Queue-sweep completion may reenter service shutdown. The admission
            // lock is released first.
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
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            let error =
                service_overloaded_error("queue_wait_elapsed", state.queued.len(), self.queue_bounds.max_waiters);
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
                self.queue_bounds.max_waiters,
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
        } else if request.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            let error =
                service_overloaded_error("queue_wait_elapsed", state.queued.len(), self.queue_bounds.max_waiters);
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
        self.notify_admission_available();
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
}

fn attempt_admission_claim_error(error: AttemptLifecycleError) -> QueryError {
    let query_error = match error {
        AttemptLifecycleError::AdmissionAlreadyClaimed => QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.query_service.attempt_admission_claim",
            "the attempt already used its one admission",
        )
        .with_fact("reason", "attempt_admission_already_claimed"),
        _ => QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.query_service.attempt_admission_claim",
            "the attempt lifecycle rejected its admission claim",
        )
        .with_fact("reason", "attempt_admission_claim_rejected"),
    };
    query_error.with_source(error)
}
