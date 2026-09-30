//! Waiter-based and async admission entry points.

use query_planning::error::Result;
use tokio::time;

use super::{
    super::{
        super::{
            admission::{AdmissionRequest, QueryAdmissionWaiter, QueuedAdmissionTerminal},
            admission_rejection,
            handle::QueryHandle,
            record_admission_lifecycle,
        },
        QueryService,
        api::QueryAdmissionRequest,
    },
    ADMISSION_CANCELLATION_POLL_INTERVAL, PreparedSubmission,
};
use crate::runtime::{
    backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
    backend_lifecycle::ObservationPermit,
};

impl QueryService {
    /// Enqueues a query admission request without blocking the caller.
    ///
    /// Call [`QueryAdmissionWaiter::try_admit`] until it returns a handle. If
    /// the waiter is dropped first, the queued request is removed.
    #[cfg(test)]
    pub(crate) fn enqueue_admission(&self) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request(QueryAdmissionRequest::new())
    }

    /// Admits a query asynchronously without parking a blocking thread.
    ///
    /// This is the async counterpart to [`Self::admit_query`]. It uses the
    /// same queue, timeout, and overload error taxonomy, but waits on an
    /// event-driven wakeup instead of polling
    /// [`QueryAdmissionWaiter::try_admit`].
    #[cfg(test)]
    pub(crate) async fn admit_async(&self) -> Result<QueryHandle> {
        self.admit_async_with_request(QueryAdmissionRequest::new()).await
    }

    /// Admits a query asynchronously with consolidated request-scoped admission metadata.
    #[cfg(test)]
    pub(crate) async fn admit_async_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        self.admit_async_with_request_and_permits(request, None, None, None).await
    }

    pub(crate) async fn admit_async_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryHandle> {
        self.admit_async_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
        .await
    }

    async fn admit_async_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryHandle> {
        let mut waiter = self.enqueue_admission_with_request_and_permits(
            request,
            observation_permit,
            gpu_resource_permit,
            completion_dispatcher,
        )?;
        loop {
            let notified = self.inner.admission_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(handle) = waiter.try_admit()? {
                return Ok(handle);
            }
            let remaining_wait = waiter.remaining_wait();
            if remaining_wait.is_zero() {
                continue;
            }
            let _ = time::timeout(remaining_wait.min(ADMISSION_CANCELLATION_POLL_INTERVAL), &mut notified).await;
        }
    }

    /// Enqueues a query admission request with consolidated request-scoped metadata.
    #[cfg(test)]
    pub(crate) fn enqueue_admission_with_request(
        &self,
        request: QueryAdmissionRequest,
    ) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request_and_permits(request, None, None, None)
    }

    fn enqueue_admission_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
    ) -> Result<QueryAdmissionWaiter> {
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
            execution_cancellation: execution_cancellation.clone(),
            terminal: QueuedAdmissionTerminal::default(),
        };
        let terminal = request.terminal.clone();
        {
            let _state = self.convert_pending_preflight_to_waiter(request, pending)?;
        }
        Ok(QueryAdmissionWaiter {
            service: self.clone(),
            admission_request_id,
            observation,
            requested_at,
            deadline,
            execution_cancellation,
            terminal,
            queued: true,
        })
    }
}
