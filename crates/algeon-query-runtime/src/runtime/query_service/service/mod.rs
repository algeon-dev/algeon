//! Bounded admission, placement, and query-service orchestration.

use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::{
    sync::mpsc::{Receiver, SyncSender},
    time::Duration,
};

#[cfg(feature = "qualification")]
use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{config::QueryExecutionConfig, controller::AdmissionController, resources::QueryServiceResources};
#[cfg(feature = "qualification")]
use crate::runtime::gpu::DeviceAdmissionHealth;
use crate::runtime::{allocator_correlation::AllocatorEventCorrelationIndex, context::QueryServiceEpoch};

mod api;
mod build;
pub(in crate::runtime::query_service) mod disposition;
mod reporting;
mod requests;
mod shutdown;
pub(in crate::runtime::query_service) mod transaction;
pub(in crate::runtime::query_service) use api::AdmissionObservation;
#[cfg(feature = "qualification")]
pub use api::DeviceQuarantineInjection;
#[cfg(test)]
pub(crate) use api::QueryAdmissionOutcome;
pub use api::{
    QueryActivityLease, QueryAdmissionRequest, QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot,
};

/// Service-level entry point for native query admission.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct QueryService {
    pub(in crate::runtime::query_service) inner: Arc<QueryServiceInner>,
}

#[derive(Debug)]
pub(in crate::runtime::query_service) struct QueryServiceInner {
    pub(in crate::runtime::query_service) config: QueryExecutionConfig,
    pub(in crate::runtime::query_service) admission: Arc<AdmissionController>,
    pub(in crate::runtime::query_service) resources: QueryServiceResources,
    pub(in crate::runtime::query_service) service_epoch: QueryServiceEpoch,
    /// Bounded domain-to-attempt state used only after allocator callbacks
    /// return. Callback writers never acquire this engine lock.
    pub(in crate::runtime::query_service) allocator_event_correlation: Mutex<AllocatorEventCorrelationIndex>,
}

/// Test-side half of a one-shot pause rendezvous.
///
/// Dropping the handle releases a paused worker so a failing assertion cannot
/// strand a test thread.
#[cfg(test)]
pub(in crate::runtime::query_service) struct TestPauseHandle {
    name: &'static str,
    entered: Mutex<Receiver<()>>,
    release: Mutex<Option<SyncSender<()>>>,
}

#[cfg(test)]
impl std::fmt::Debug for TestPauseHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TestPauseHandle").field("name", &self.name).finish_non_exhaustive()
    }
}

#[cfg(test)]
impl TestPauseHandle {
    pub(in crate::runtime::query_service) fn new(
        name: &'static str,
        entered: Receiver<()>,
        release: SyncSender<()>,
    ) -> Self {
        Self { name, entered: Mutex::new(entered), release: Mutex::new(Some(release)) }
    }

    pub(in crate::runtime::query_service) fn wait_until_entered(&self, timeout: Duration) {
        self.entered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv_timeout(timeout)
            .unwrap_or_else(|error| panic!("{} did not enter before the deadline: {error}", self.name));
    }

    pub(in crate::runtime::query_service) fn release(&self) {
        let release = self.release.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(release) = release
            && release.send(()).is_err()
        {
            panic!("paused {} remains live until release", self.name);
        }
    }
}

#[cfg(test)]
impl Drop for TestPauseHandle {
    fn drop(&mut self) {
        if let Some(release) = self.release.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            let _ = release.try_send(());
        }
    }
}

impl QueryService {
    /// Performs an explicit, controlled device-local quarantine transition.
    ///
    /// This is intentionally an engine capability rather than a backend
    /// configuration mode. It owns the same ledger health state consulted by
    /// placement, takes the fleet admission gate used by placement, and leaves
    /// native cuDF health untouched. A caller can therefore validate fallback
    /// after a quarantine without claiming to have produced a real CUDA fault.
    #[cfg(feature = "qualification")]
    pub(crate) fn inject_device_quarantine(&self, device_ordinal: usize) -> Result<DeviceQuarantineInjection> {
        const REASON: &str = "controlled validation device quarantine";

        let injection = {
            let _fleet_admission = self.inner.admission.fleet_admission.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.fault_injection",
                    "fleet admission gate was poisoned",
                )
                .with_fact("reason", "fleet_admission_gate_poisoned")
            })?;
            let ledger = self.inner.admission.devices.ledger(device_ordinal)?;
            let newly_quarantined = ledger.close_admission_if_open(REASON)?;
            let DeviceAdmissionHealth::Closed { generation, .. } = ledger.health()? else {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.query_service.fault_injection",
                    "device ledger remained open after controlled quarantine",
                )
                .with_fact("reason", "quarantine_transition_not_observed")
                .with_fact("device_ordinal", device_ordinal));
            };
            DeviceQuarantineInjection::new(device_ordinal, generation, newly_quarantined)
        };
        self.inner.admission.notify_admission_available();
        Ok(injection)
    }
}
