//! Admission entry points. Each enqueues one [`QueryAdmissionWaiter`] and
//! drives it to an admitted handle or a structured rejection.

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
    PreparedSubmission,
};
use crate::runtime::{
    backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
    backend_lifecycle::ObservationPermit,
};

impl QueryService {
    #[cfg(test)]
    pub(crate) fn admit_query(&self) -> Result<QueryHandle> {
        self.admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    /// Admits a query on the calling thread, which parks until the request
    /// is admitted or rejected.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        self.enqueue_admission_with_request_and_permits(request, None, None, None)?.admit_blocking()
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn admit_query_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryHandle> {
        self.enqueue_admission_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )?
        .admit_blocking()
    }

    /// Enqueues a query admission request without blocking the caller.
    ///
    /// Call [`QueryAdmissionWaiter::try_admit`] until it returns a handle. If
    /// the waiter is dropped first, the queued request is removed.
    #[cfg(test)]
    pub(crate) fn enqueue_admission(&self) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request(QueryAdmissionRequest::test_attempt())
    }

    /// Admits a query asynchronously without parking a blocking thread.
    #[cfg(test)]
    pub(crate) async fn admit_async(&self) -> Result<QueryHandle> {
        self.admit_async_with_request(QueryAdmissionRequest::test_attempt()).await
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
            if let Some(handle) = waiter.try_admit_async().await? {
                return Ok(handle);
            }
            let remaining_wait = waiter.remaining_wait();
            if remaining_wait.is_zero() {
                continue;
            }
            // `try_admit` reports cancellation and the queue deadline; each
            // of them, like a ledger transition, ends this wait.
            let _ = time::timeout(remaining_wait, async {
                tokio::select! {
                    () = &mut notified => {},
                    () = waiter.execution_cancellation.cancelled_or_expired() => {},
                }
            })
            .await;
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

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn enqueue_admission_with_backend_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: ObservationPermit,
        gpu_resource_permit: GpuResourcePermit,
        completion_dispatcher: ObservationCompletionDispatcher,
    ) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request_and_permits(
            request,
            Some(observation_permit),
            Some(gpu_resource_permit),
            Some(completion_dispatcher),
        )
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
            observation,
            attempt_id,
            admission_request_id,
            completion,
            requirement,
            required_capabilities,
            candidate_ordinals,
        } = prepared;
        let deadline = self.inner.queue_bounds.wait_deadline(requested_at, queue_deadline);
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

        let (device_entries, compatible_devices) = match self.resolve_device_admission_entries(
            submitted_plan_capability_evidence.as_ref(),
            &requirement,
            &candidate_ordinals,
        ) {
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
            device_state_validated: false,
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
