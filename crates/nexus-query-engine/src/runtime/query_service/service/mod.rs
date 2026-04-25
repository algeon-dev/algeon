//! Tenant admission, placement, and query-service orchestration.

use super::*;

mod adaptive;
mod api;
mod build;
mod grant_policy;
mod placement;
mod queue;
mod reporting;
mod requests;

pub use api::{
    QueryActivityLease, QueryAdmissionRequest, QueryGrantBand, QueryGrantRequest, QueryServiceAdmissionSnapshot,
    QueryServiceDeviceAdmissionSnapshot, QueryServiceGraphCacheSnapshot, TenantQueueDepth,
};
pub(in crate::runtime::query_service) use grant_policy::TenantGrantPolicy;
pub(in crate::runtime::query_service::service) use queue::query_service_shutdown_timeout;

/// Service-level entry point for tenant-aware native query admission.
#[derive(Clone, Debug)]
pub struct QueryService {
    pub(in crate::runtime::query_service) inner: Arc<QueryServiceInner>,
}

#[derive(Debug)]
pub(in crate::runtime::query_service) struct QueryServiceInner {
    pub(in crate::runtime::query_service) config: QueryServiceConfig,
    pub(in crate::runtime::query_service) tenant_grants: HashMap<TenantId, TenantGrantPolicy>,
    pub(in crate::runtime::query_service) tenant_order: Vec<TenantId>,
    pub(in crate::runtime::query_service) max_tenant_min_bytes: u64,
    pub(in crate::runtime::query_service) fallback_memory_governor: MemoryGovernor,
    pub(in crate::runtime::query_service) resources: Option<QueryServiceResources>,
    pub(in crate::runtime::query_service) service_epoch: QueryServiceEpoch,
    pub(in crate::runtime::query_service) admission_open: AtomicBool,
    pub(in crate::runtime::query_service) admission: Mutex<AdmissionState>,
    pub(in crate::runtime::query_service) admission_available: Condvar,
    pub(in crate::runtime::query_service) admission_notify: Notify,
    #[cfg(test)]
    pub(in crate::runtime::query_service) shutdown_waiters: AtomicUsize,
    pub(in crate::runtime::query_service) next_admission_request_id: AtomicU64,
    pub(in crate::runtime::query_service) next_admission_sequence: AtomicU64,
}

impl QueryServiceInner {
    pub(in crate::runtime::query_service) fn notify_admission_available(&self) {
        self.admission_available.notify_all();
        self.admission_notify.notify_waiters();
    }
}

impl QueryService {
    /// Returns true when both handles point at the same service instance.
    pub fn same_service(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
