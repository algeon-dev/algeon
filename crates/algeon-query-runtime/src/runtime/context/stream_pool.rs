//! Epoch-guarded access to a service stream pool.

use std::sync::Arc;

use query_planning::error::Result;

use super::{
    super::{ServiceStreamPool, StreamLease},
    ExecutionCancellation, QueryResourceAnchor, QueryServiceEpoch, QueryServiceResourceEpoch,
};
use crate::observability::{RuntimeObservationHandle, RuntimeStreamLeaseScope};

/// Revocable context-scoped access to a service stream pool.
///
/// Query-service handles retain their query admission anchor even before a
/// stream is acquired. Each acquisition atomically checks both the query and
/// service epochs and gives the returned stream lease a counted owner that
/// survives derived runtime clones.
#[derive(Debug, Clone)]
pub struct QueryServiceStreamPool {
    pub(super) pool: ServiceStreamPool,
    pub(super) service_epoch: Option<QueryServiceEpoch>,
    pub(super) query_anchor: Option<Arc<QueryResourceAnchor>>,
}

impl QueryServiceStreamPool {
    pub fn device(&self) -> cudf::DeviceId {
        self.pool.device()
    }

    pub fn pool_size(&self) -> usize {
        self.pool.pool_size()
    }

    pub fn acquire(&self) -> Result<StreamLease> {
        let operation = "query_engine.runtime.context.service_stream_pool.acquire";
        let resource_epoch = self.acquire_resource_epoch(operation)?;
        let lease = self.pool.acquire()?;
        if let Some(epoch) = &resource_epoch {
            epoch.check_open(operation)?;
        }
        Ok(self.attach_resource_epoch(lease, resource_epoch))
    }

    pub(super) fn acquire_observed_with_cancellation(
        &self,
        cancellation: &ExecutionCancellation,
        operation: &'static str,
        observation: RuntimeObservationHandle,
        scope: RuntimeStreamLeaseScope,
    ) -> Result<StreamLease> {
        let resource_epoch = self.acquire_resource_epoch(operation)?;
        let lease = self.pool.acquire_observed_with_cancellation(cancellation, operation, observation, scope)?;
        if let Some(epoch) = &resource_epoch {
            epoch.check_open(operation)?;
        }
        Ok(self.attach_resource_epoch(lease, resource_epoch))
    }

    fn acquire_resource_epoch(&self, operation: &'static str) -> Result<Option<Arc<QueryServiceResourceEpoch>>> {
        self.service_epoch
            .as_ref()
            .map(|epoch| QueryServiceResourceEpoch::acquire(epoch, self.query_anchor.as_ref(), operation))
            .transpose()
    }

    fn attach_resource_epoch(
        &self,
        lease: StreamLease,
        resource_epoch: Option<Arc<QueryServiceResourceEpoch>>,
    ) -> StreamLease {
        match resource_epoch {
            Some(resource_epoch) => lease.with_resource_epoch(resource_epoch),
            None => lease,
        }
    }
}
