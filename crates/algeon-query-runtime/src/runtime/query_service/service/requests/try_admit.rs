//! Non-blocking admission entry points.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use query_planning::error::Result;

use super::{
    super::{
        super::{
            admission::{DeviceAdmissionEntries, admission_cancellation_or_deadline_error},
            admission_rejection,
            device_waiters::PendingDeviceSlotNotifications,
            handle::QueryHandle,
            record_admission_lifecycle,
        },
        QueryService,
        api::QueryAdmissionRequest,
    },
    PlacedAdmissionCommit, PreparedSubmission,
};
use crate::{
    observability::AdmissionLifecycleState,
    runtime::{
        backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
        backend_lifecycle::ObservationPermit,
    },
};

impl QueryService {
    /// Attempts to admit a query without queueing or waiting.
    ///
    /// Returns `Ok(None)` when all admission slots are active or when queued
    /// requests already exist and this caller must not bypass them.
    #[cfg(test)]
    pub fn try_admit_query(&self) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request(QueryAdmissionRequest::new())
    }

    /// Attempts to admit a query with consolidated request-scoped metadata.
    #[cfg(test)]
    pub fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request_and_permits(request, None, None, None)
    }

    #[doc(hidden)]
    pub fn try_admit_query_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
    }

    fn try_admit_query_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<Option<QueryHandle>> {
        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let prepared =
            self.prepare_submission(request, observation_permit, gpu_resource_permit, completion_dispatcher)?;
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
                &prepared.observation,
                prepared.admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            // Completion may reenter service shutdown, so release the admission lock first.
            drop(state);
            prepared.completion.finish_error(&error);
            return Err(error);
        }
        if prepared.execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(&prepared.execution_cancellation);
            record_admission_lifecycle(
                &prepared.observation,
                prepared.admission_request_id,
                AdmissionLifecycleState::CancellationRequested,
                state.queued.len(),
                None,
                None,
            );
            record_admission_lifecycle(
                &prepared.observation,
                prepared.admission_request_id,
                AdmissionLifecycleState::Cancelled,
                state.queued.len(),
                None,
                None,
            );
            // Completion may reenter service shutdown, so release the admission lock first.
            drop(state);
            prepared.completion.finish_error(&error);
            return Err(error);
        }
        if !state.queued.is_empty() || !state.pending_preflight.is_empty() {
            return Ok(None);
        }
        // Device-specific owner preflight must not extend the admission-lock window.
        drop(state);

        let (device_entries, compatible_devices) = match self.resolve_device_admission_entries(
            &prepared.request,
            &prepared.requirement,
            &prepared.candidate_ordinals,
        ) {
            Ok(entries) => entries,
            Err(error) => {
                record_admission_lifecycle(
                    &prepared.observation,
                    prepared.admission_request_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                prepared.completion.finish_error(&error);
                return Err(error);
            },
        };
        if let Err(error) = self.reject_impossible_requirements(
            &device_entries,
            &prepared.required_capabilities,
            prepared.requested_query_cap_bytes,
        ) {
            record_admission_lifecycle(
                &prepared.observation,
                prepared.admission_request_id,
                admission_rejection(&error),
                0,
                None,
                None,
            );
            prepared.completion.finish_error(&error);
            return Err(error);
        }

        self.place_try_admit_request(prepared, &device_entries, compatible_devices, &mut slot_notifications)
    }

    fn place_try_admit_request(
        &self,
        prepared: PreparedSubmission,
        device_entries: &DeviceAdmissionEntries,
        compatible_devices: Vec<usize>,
        slot_notifications: &mut PendingDeviceSlotNotifications,
    ) -> Result<Option<QueryHandle>> {
        let PreparedSubmission {
            observation,
            attempt_id,
            admission_request_id,
            completion,
            requested_query_cap_bytes,
            submitted_plan_capability_evidence,
            queue_deadline,
            execution_cancellation,
            ..
        } = prepared;
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
                &observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            // Completion may reenter service shutdown, so release the admission lock first.
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        if !state.queued.is_empty() || !state.pending_preflight.is_empty() {
            return Ok(None);
        }
        if execution_cancellation.is_cancelled() {
            let error = admission_cancellation_or_deadline_error(&execution_cancellation);
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
            // Completion may reenter service shutdown, so release the admission lock first.
            drop(state);
            completion.finish_error(&error);
            return Err(error);
        }
        let placed_query = match self.try_place_query(
            admission_request_id,
            attempt_id,
            device_entries,
            &compatible_devices,
            requested_query_cap_bytes,
            &mut *slot_notifications,
        ) {
            Ok(placed_query) => placed_query,
            Err(error) => {
                record_admission_lifecycle(
                    &observation,
                    admission_request_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                // Completion may reenter service shutdown, so release the admission lock first.
                drop(state);
                completion.finish_error(&error);
                return Err(error);
            },
        };
        let Some(placed_query) = placed_query else {
            return Ok(None);
        };
        if let Some(error) = self.admission_terminal_rejection(&state) {
            record_admission_lifecycle(
                &observation,
                admission_request_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            // Grant cleanup and completion may reenter shutdown, so release the admission lock first.
            drop(state);
            drop(placed_query);
            completion.finish_error(&error);
            return Err(error);
        }
        let lifecycle_ticket = state.lifecycle.begin_provisional(Arc::downgrade(&self.inner));
        self.inner.notify_admission_available();
        // Admission commit may wait for resource handoff, so release the controller lock first.
        drop(state);
        let configured_deadline =
            Instant::now().checked_add(self.inner.queue_bounds.max_wait).unwrap_or_else(Instant::now);
        let deadline = queue_deadline.map_or(configured_deadline, |deadline| deadline.min(configured_deadline));
        self.commit_placed_admission(PlacedAdmissionCommit {
            placed_query,
            admitted_device_requirements: device_entries
                .iter()
                .map(|(&ordinal, entry)| (ordinal, entry.requirement.clone()))
                .collect(),
            submitted_plan_capability_evidence,
            admitted_compatible_devices: compatible_devices,
            observation,
            attempt_id,
            completion,
            admission_request_id,
            admission_wait: Duration::ZERO,
            deadline,
            execution_cancellation,
            admission_wait_fact: None,
            lifecycle_ticket,
        })
        .map(Some)
    }
}
