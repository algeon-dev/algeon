//! Admission entry points. Each submits one request to the admission
//! controller and drives its waiter to an admitted handle or a structured
//! rejection.

use std::sync::Arc;

use query_planning::error::Result;

#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::query_service::admission::QueryAdmissionWaiter;
use crate::runtime::{
    backend_core::{GpuResourcePermit, ObservationCompletionDispatcher},
    backend_lifecycle::ObservationPermit,
    query_service::{
        handle::QueryHandle,
        service::{QueryService, api::QueryAdmissionRequest, transaction::AdmissionTransactionProbe},
    },
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
        self.enqueue_admission_with_request_and_permits(request, None, None, None, None)?.admit_blocking()
    }

    /// Admits a query on the calling thread with a fault-injection probe over
    /// its admission transaction.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn admit_query_with_probe(
        &self,
        request: QueryAdmissionRequest,
        probe: Arc<AdmissionTransactionProbe>,
    ) -> Result<QueryHandle> {
        self.enqueue_admission_with_request_and_permits(request, None, None, None, Some(probe))?.admit_blocking()
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
            None,
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
        self.admit_async_with_request_and_permits(request, None, None, None, None).await
    }

    /// Admits a query asynchronously with a fault-injection probe over its
    /// admission transaction.
    #[cfg(test)]
    pub(in crate::runtime::query_service) async fn admit_async_with_probe(
        &self,
        request: QueryAdmissionRequest,
        probe: Arc<AdmissionTransactionProbe>,
    ) -> Result<QueryHandle> {
        self.admit_async_with_request_and_permits(request, None, None, None, Some(probe)).await
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
            None,
        )
        .await
    }

    /// The one async admission path: enqueue, wait for placement on the host
    /// runtime, then build and publish the envelope on the device pool.
    ///
    /// The permits are `None` only for a service used without a backend
    /// (tests and embedders); a backend-admitted request always carries both,
    /// and they stay with the request's completion whatever the outcome.
    async fn admit_async_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
        probe: Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<QueryHandle> {
        #[cfg(any(test, feature = "test-hooks"))]
        let (request, test_attempt) = request.split_test_attempt();
        let mut waiter =
            self.inner.admission.enqueue(request, observation_permit, gpu_resource_permit, completion_dispatcher)?;
        // A caller that stops waiting drops the placed admission: its ledger
        // grant and lifecycle ticket roll back, and the completion recovers
        // the attempt's terminal outcome.
        let placed = waiter.place_async().await?;
        let handle = self.commit_placed_admission_async(placed, probe).await?;
        #[cfg(any(test, feature = "test-hooks"))]
        handle.own_test_attempt(test_attempt);
        Ok(handle)
    }

    /// Enqueues a query admission request with consolidated request-scoped metadata.
    #[cfg(test)]
    pub(crate) fn enqueue_admission_with_request(
        &self,
        request: QueryAdmissionRequest,
    ) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_request_and_permits(request, None, None, None, None)
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
            None,
        )
    }

    #[cfg(any(test, feature = "test-hooks"))]
    fn enqueue_admission_with_request_and_permits(
        &self,
        request: QueryAdmissionRequest,
        observation_permit: Option<ObservationPermit>,
        gpu_resource_permit: Option<GpuResourcePermit>,
        completion_dispatcher: Option<ObservationCompletionDispatcher>,
        probe: Option<Arc<AdmissionTransactionProbe>>,
    ) -> Result<QueryAdmissionWaiter> {
        let (request, test_attempt) = request.split_test_attempt();
        let waiter =
            self.inner.admission.enqueue(request, observation_permit, gpu_resource_permit, completion_dispatcher)?;
        Ok(QueryAdmissionWaiter { service: self.clone(), waiter, test_attempt, probe })
    }
}
