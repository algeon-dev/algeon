//! Blocking queue-backed admission entry points.

use std::{
    sync::{Arc, MutexGuard},
    time::Instant,
};

use query_planning::error::Result;

use super::{
    super::{
        super::{
            admission::{
                AdmissionRequest, AdmissionState, PlacedQuery, QueuedAdmissionTerminal, finish_removed_request_error,
                service_overloaded_error,
            },
            admission_rejection,
            device_waiters::PendingDeviceSlotNotifications,
            handle::QueryHandle,
            record_admission_lifecycle,
        },
        QueryService,
        api::QueryAdmissionRequest,
    },
    ADMISSION_CANCELLATION_POLL_INTERVAL, PlacedAdmissionCommit, PreparedSubmission,
};
use crate::{
    observability::{AdmissionLifecycleState, AdmissionRequestId, RuntimeObservationHandle},
    runtime::{
        backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
        backend_lifecycle::ObservationPermit,
    },
};

struct QueuedAdmissionAttempt<'a> {
    observation: &'a RuntimeObservationHandle,
    admission_request_id: AdmissionRequestId,
    requested_at: Instant,
    deadline: Instant,
    slot_notifications: &'a mut PendingDeviceSlotNotifications,
}

impl QueryService {
    #[cfg(test)]
    pub fn admit_query(&self) -> Result<QueryHandle> {
        self.admit_query_with_request(QueryAdmissionRequest::new())
    }

    /// Admits a query with consolidated request-scoped admission metadata.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        self.admit_query_with_request_and_permits(request, None, None, None)
    }

    #[doc(hidden)]
    pub fn admit_query_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryHandle> {
        self.admit_query_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
    }

    fn admit_query_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryHandle> {
        let prepared =
            self.prepare_submission(request, observation_permit, gpu_resource_permit, completion_dispatcher)?;
        let PreparedSubmission {
            queue_deadline,
            requested_query_cap_bytes,
            requested_at,
            execution_cancellation,
            submitted_plan_capability_evidence,
            request,
            observation,
            attempt_id,
            admission_request_id,
            completion,
            requirement,
            required_capabilities,
            candidate_ordinals,
        } = prepared;
        let configured_deadline = requested_at.checked_add(self.inner.queue_bounds.max_wait).unwrap_or(requested_at);
        let deadline = queue_deadline.map_or(configured_deadline, |deadline| deadline.min(configured_deadline));
        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let pending = match self.reserve_pending_preflight_for_request(
            &observation,
            admission_request_id,
            &execution_cancellation,
            deadline,
        ) {
            Ok(pending) => pending,
            Err(error) => {
                completion.finish_error(&error);
                return Err(error);
            },
        };
        let submission_seq = pending.submission_seq();

        let (device_entries, compatible_devices) =
            match self.resolve_device_admission_entries(&request, &requirement, &candidate_ordinals) {
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
            execution_cancellation,
            terminal: QueuedAdmissionTerminal::default(),
        };
        let terminal = request.terminal.clone();
        let state = self.convert_pending_preflight_to_waiter(request, pending)?;
        let mut attempt = QueuedAdmissionAttempt {
            observation: &observation,
            admission_request_id,
            requested_at,
            deadline,
            slot_notifications: &mut slot_notifications,
        };
        self.drive_queued_admission(&mut attempt, &terminal, state)
    }
    fn place_selected_queued_request<'a>(
        &'a self,
        mut state: MutexGuard<'a, AdmissionState>,
        position: usize,
        device_ordinal: usize,
        attempt: &mut QueuedAdmissionAttempt<'_>,
    ) -> Result<(MutexGuard<'a, AdmissionState>, Option<PlacedQuery>)> {
        let selected_requirement = state.queued[position].requirement_for_device(device_ordinal)?.clone();
        let placed_query = match self.try_place_on_device(
            device_ordinal,
            attempt.admission_request_id,
            state.queued[position].attempt_id,
            &selected_requirement,
            state.queued[position].requested_query_cap_bytes,
            &mut *attempt.slot_notifications,
        ) {
            Ok(placed_query) => placed_query,
            Err(error) => {
                let removed = self.abandon_queued_request_locked(&mut state, attempt.admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        attempt.observation,
                        attempt.admission_request_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            },
        };
        Ok((state, placed_query))
    }
    fn wait_for_queued_placement<'a>(
        &'a self,
        mut state: MutexGuard<'a, AdmissionState>,
        position: usize,
        device_ordinal: usize,
        attempt: &mut QueuedAdmissionAttempt<'_>,
    ) -> Result<MutexGuard<'a, AdmissionState>> {
        if attempt.slot_notifications.has_pending() {
            drop(state);
            attempt.slot_notifications.notify();
            state = match self.inner.admission.lock() {
                Ok(state) => state,
                Err(error) => {
                    return Err(self.recover_queued_request_after_admission_poison(
                        error.into_inner(),
                        attempt.admission_request_id,
                    ));
                },
            };
            if self.admission_is_closed(&state) {
                return Ok(state);
            }
        }
        let now = Instant::now();
        if now >= attempt.deadline {
            let requested_query_cap_bytes = state.queued[position].requested_query_cap_bytes;
            let removed = self.abandon_queued_request_locked(&mut state, attempt.admission_request_id);
            if removed.is_some() {
                record_admission_lifecycle(
                    attempt.observation,
                    attempt.admission_request_id,
                    AdmissionLifecycleState::TimedOut,
                    state.queued.len(),
                    None,
                    None,
                );
            }
            let error =
                self.memory_grant_wait_timeout_error(device_ordinal, requested_query_cap_bytes, state.queued.len());
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }
        let wait = attempt.deadline.saturating_duration_since(now).min(ADMISSION_CANCELLATION_POLL_INTERVAL);
        let (next_state, timeout) = match self.inner.admission_available.wait_timeout(state, wait) {
            Ok(waited) => waited,
            Err(error) => {
                return Err(self.recover_queued_request_after_admission_poison(
                    error.into_inner().0,
                    attempt.admission_request_id,
                ));
            },
        };
        state = next_state;
        if timeout.timed_out() && Instant::now() >= attempt.deadline {
            let requested_query_cap_bytes = state.queued[position].requested_query_cap_bytes;
            let removed = self.abandon_queued_request_locked(&mut state, attempt.admission_request_id);
            if removed.is_some() {
                record_admission_lifecycle(
                    attempt.observation,
                    attempt.admission_request_id,
                    AdmissionLifecycleState::TimedOut,
                    state.queued.len(),
                    None,
                    None,
                );
            }
            let error =
                self.memory_grant_wait_timeout_error(device_ordinal, requested_query_cap_bytes, state.queued.len());
            drop(state);
            finish_removed_request_error(removed, &error);
            return Err(error);
        }
        Ok(state)
    }

    fn commit_selected_placement(
        &self,
        mut state: MutexGuard<'_, AdmissionState>,
        position: usize,
        placed_query: PlacedQuery,
        attempt: &QueuedAdmissionAttempt<'_>,
    ) -> Result<QueryHandle> {
        let admitted = state.remove_queued_at(position).expect("admissible request exists");
        let admission_wait_fact = admitted.admission_wait_fact();
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                &admitted.observation,
                attempt.admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            drop(state);
            drop(placed_query);
            finish_removed_request_error(Some(admitted), &error);
            return Err(error);
        }
        let lifecycle_ticket = state.lifecycle.begin_provisional(Arc::downgrade(&self.inner));
        self.inner.notify_admission_available();
        drop(state);
        self.commit_placed_admission(PlacedAdmissionCommit {
            placed_query,
            admitted_device_requirements: admitted
                .device_entries
                .iter()
                .map(|(&ordinal, requirement)| (ordinal, requirement.clone()))
                .collect(),
            submitted_plan_capability_evidence: admitted.submitted_plan_capability_evidence.clone(),
            admitted_compatible_devices: admitted.compatible_devices.clone(),
            observation: admitted.observation,
            attempt_id: admitted.attempt_id,
            completion: admitted.completion,
            admission_request_id: attempt.admission_request_id,
            admission_wait: attempt.requested_at.elapsed(),
            deadline: attempt.deadline,
            execution_cancellation: admitted.execution_cancellation,
            admission_wait_fact: Some(admission_wait_fact),
            lifecycle_ticket,
        })
    }

    fn drive_queued_admission<'a>(
        &'a self,
        attempt: &mut QueuedAdmissionAttempt<'_>,
        terminal: &QueuedAdmissionTerminal,
        mut state: MutexGuard<'a, AdmissionState>,
    ) -> Result<QueryHandle> {
        loop {
            if let Some(error) = terminal.take() {
                return Err(error);
            }
            let sweep = self.sweep_expired_queued_requests_locked(&mut state);
            if !sweep.is_empty() {
                // Queue-sweep completion may reenter service shutdown, so release the admission lock first.
                drop(state);
                sweep.finish();
                state = match self.inner.admission.lock() {
                    Ok(state) => state,
                    Err(error) => {
                        return Err(self.recover_queued_request_after_admission_poison(
                            error.into_inner(),
                            attempt.admission_request_id,
                        ));
                    },
                };
                continue;
            }
            if let Some(error) = self.admission_terminal_rejection(&state) {
                let removed =
                    self.remove_queued_request_without_admission_locked(&mut state, attempt.admission_request_id);
                self.inner.notify_admission_available();
                if removed.is_some() {
                    record_admission_lifecycle(
                        attempt.observation,
                        attempt.admission_request_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            }
            let selected = match self.select_next_runnable_locked(&mut state) {
                Ok(selected) => match selected.filter(|(position, _)| {
                    state.queued[*position].admission_request_id == attempt.admission_request_id
                }) {
                    Some(selected) => Ok(Some(selected)),
                    None => self.fallback_driver_selection_locked(&state, attempt.admission_request_id),
                },
                Err(error) => Err(error),
            };
            let selected = match selected {
                Ok(selected) => selected,
                Err(error) => {
                    let removed = self.abandon_queued_request_locked(&mut state, attempt.admission_request_id);
                    if removed.is_some() {
                        record_admission_lifecycle(
                            attempt.observation,
                            attempt.admission_request_id,
                            admission_rejection(&error),
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    drop(state);
                    finish_removed_request_error(removed, &error);
                    return Err(error);
                },
            };
            if let Some((position, device_ordinal)) = selected {
                let (next_state, placed_query) =
                    self.place_selected_queued_request(state, position, device_ordinal, &mut *attempt)?;
                state = next_state;
                if let Some(placed_query) = placed_query {
                    return self.commit_selected_placement(state, position, placed_query, &*attempt);
                }
                state = self.wait_for_queued_placement(state, position, device_ordinal, &mut *attempt)?;
                continue;
            }
            let now = Instant::now();
            if now >= attempt.deadline {
                let removed = self.abandon_queued_request_locked(&mut state, attempt.admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        attempt.observation,
                        attempt.admission_request_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                let error = service_overloaded_error(
                    "queue_wait_elapsed",
                    state.queued.len(),
                    self.inner.queue_bounds.max_waiters,
                );
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            }
            let wait = attempt.deadline.saturating_duration_since(now).min(ADMISSION_CANCELLATION_POLL_INTERVAL);
            let (next_state, timeout) = match self.inner.admission_available.wait_timeout(state, wait) {
                Ok(waited) => waited,
                Err(error) => {
                    return Err(self.recover_queued_request_after_admission_poison(
                        error.into_inner().0,
                        attempt.admission_request_id,
                    ));
                },
            };
            state = next_state;
            if timeout.timed_out() && Instant::now() >= attempt.deadline {
                let removed = self.abandon_queued_request_locked(&mut state, attempt.admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        attempt.observation,
                        attempt.admission_request_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                let error = service_overloaded_error(
                    "queue_wait_elapsed",
                    state.queued.len(),
                    self.inner.queue_bounds.max_waiters,
                );
                drop(state);
                finish_removed_request_error(removed, &error);
                return Err(error);
            }
        }
    }
}
