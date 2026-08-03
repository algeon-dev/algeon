//! Bounded admission, placement, and query-service orchestration.

use super::*;
use crate::runtime::allocator_correlation::AllocatorEventCorrelationIndex;

mod api;
mod build;
pub(in crate::runtime::query_service) mod disposition;
mod placement;
mod queue;
mod reporting;
mod requests;
pub(in crate::runtime::query_service) mod transaction;

pub use api::{
    QueryActivityLease, QueryAdmissionRequest, QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot,
};
pub(in crate::runtime::query_service::service) use queue::query_service_shutdown_timeout;

/// Service-level entry point for native query admission.
#[derive(Clone, Debug)]
pub(crate) struct QueryService {
    pub(in crate::runtime::query_service) inner: Arc<QueryServiceInner>,
}

#[derive(Debug)]
pub(in crate::runtime::query_service) struct QueryServiceInner {
    pub(in crate::runtime::query_service) config: QueryExecutionConfig,
    pub(in crate::runtime::query_service) queue_bounds: QueueBounds,
    pub(in crate::runtime::query_service) fallback_device_ledger: Option<DeviceLedger>,
    pub(in crate::runtime::query_service) resources: Option<QueryServiceResources>,
    pub(in crate::runtime::query_service) service_epoch: QueryServiceEpoch,
    pub(in crate::runtime::query_service) admission_open: AtomicBool,
    /// Serializes every ledger admission transaction with a fleet-wide
    /// unknown-isolation closure. The queue mutex orders controller state;
    /// this gate additionally makes the multi-ledger health transition one
    /// atomic scheduler boundary.
    pub(in crate::runtime::query_service) fleet_admission: Mutex<()>,
    pub(in crate::runtime::query_service) admission: Mutex<AdmissionState>,
    pub(in crate::runtime::query_service) admission_available: Condvar,
    pub(in crate::runtime::query_service) admission_notify: Notify,
    /// Bounded domain-to-attempt state used only after allocator callbacks
    /// return. Callback writers never acquire this engine lock.
    pub(in crate::runtime::query_service) allocator_event_correlation: Mutex<AllocatorEventCorrelationIndex>,
    #[cfg(test)]
    pub(in crate::runtime::query_service) shutdown_waiters: AtomicUsize,
    pub(in crate::runtime::query_service) admission_transaction_probe:
        Mutex<Option<Arc<transaction::AdmissionTransactionProbe>>>,
    #[cfg(test)]
    unknown_isolation_closure_pause: Mutex<Option<Arc<UnknownIsolationClosurePause>>>,
    /// One-shot fault injection after a closed protection target has released
    /// its ledger-local reservation and before the waiter can acquire a new
    /// target. This proves that the affected waiter, rather than an unrelated
    /// caller that happens to drive the next queue sweep, owns that failure.
    #[cfg(test)]
    protection_retarget_failure_after_release: AtomicBool,
    pub(in crate::runtime::query_service) next_admission_request_id: AtomicU64,
    /// Monotonic FIFO submission identity, independent of eventual admission order.
    pub(in crate::runtime::query_service) next_submission_seq: AtomicU64,
    pub(in crate::runtime::query_service) next_admission_sequence: AtomicU64,
}

impl QueryServiceInner {
    pub(in crate::runtime::query_service) fn notify_admission_available(&self) {
        self.admission_available.notify_all();
        self.admission_notify.notify_waiters();
    }

    /// Fails closed when cleanup cannot establish fault isolation.  Ledger
    /// health is the only scheduler-health authority: every affected ledger
    /// is closed before a caller can observe the terminal report or attempt a
    /// replacement admission.  In-flight grants remain valid and release in
    /// their normal cleanup path, but no closed ledger may issue another one.
    pub(in crate::runtime::query_service) fn close_unknown_isolation_admission(&self, detail: impl AsRef<str>) {
        // Placement takes this gate around its entire ledger transaction. Do
        // not close device ledgers one at a time without it: an admission
        // could otherwise race between the first and final closure and start
        // work on a fleet whose fault isolation is unknown.
        let _fleet_admission = self.fleet_admission.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let reason = format!("unknown-isolation backend failure: {}", detail.as_ref());
        let mut closed_a_ledger = false;
        if let Some(resources) = &self.resources {
            for context in resources.devices.contexts() {
                if let Err(error) = context.ledger().close_admission(reason.clone()) {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to close device admission after unknown-isolation failure: {error}"
                    );
                }
                if !closed_a_ledger {
                    closed_a_ledger = true;
                    #[cfg(test)]
                    self.pause_unknown_isolation_closure_after_first_ledger();
                }
            }
        }
        if let Some(ledger) = &self.fallback_device_ledger
            && let Err(error) = ledger.close_admission(reason)
        {
            eprintln!(
                "[nexus-query-engine] warning: failed to close fallback admission ledger after unknown-isolation failure: {error}"
            );
        }
        #[cfg(test)]
        if !closed_a_ledger && self.fallback_device_ledger.is_some() {
            self.pause_unknown_isolation_closure_after_first_ledger();
        }
        self.notify_admission_available();
    }

    #[cfg(test)]
    fn pause_unknown_isolation_closure_after_first_ledger(&self) {
        let pause =
            self.unknown_isolation_closure_pause.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(pause) = pause {
            pause.wait();
        }
    }
}

/// Test-only rendezvous after the first ledger of a fleet closure has been
/// marked closed. The closure keeps `fleet_admission` while waiting, exposing
/// the exact interleaving that must not admit work on a later device.
#[cfg(test)]
#[derive(Debug)]
struct UnknownIsolationClosurePause {
    entered: std::sync::mpsc::SyncSender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

#[cfg(test)]
impl UnknownIsolationClosurePause {
    fn wait(&self) {
        self.entered.send(()).expect("unknown-isolation closure pause receiver remains live");
        self.release
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv()
            .expect("unknown-isolation closure pause is released");
    }
}

/// Test-side half of a one-shot unknown-isolation closure pause.
#[cfg(test)]
#[derive(Debug)]
pub(in crate::runtime::query_service) struct UnknownIsolationClosurePauseHandle {
    entered: Mutex<std::sync::mpsc::Receiver<()>>,
    release: Mutex<Option<std::sync::mpsc::SyncSender<()>>>,
}

#[cfg(test)]
impl UnknownIsolationClosurePauseHandle {
    pub(in crate::runtime::query_service) fn wait_until_entered(&self, timeout: Duration) {
        self.entered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv_timeout(timeout)
            .expect("unknown-isolation closure did not close its first ledger before the deadline");
    }

    pub(in crate::runtime::query_service) fn release(&self) {
        if let Some(release) = self.release.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            release.send(()).expect("paused unknown-isolation closure remains live until release");
        }
    }
}

#[cfg(test)]
impl Drop for UnknownIsolationClosurePauseHandle {
    fn drop(&mut self) {
        if let Some(release) = self.release.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            let _ = release.send(());
        }
    }
}

impl QueryService {
    /// Returns the first configured ordinal for diagnostics and admission-only
    /// fallback paths. Resource ownership remains in ledgers/profiles rather
    /// than in execution configuration.
    pub(super) fn default_device_ordinal(&self) -> Option<usize> {
        self.inner
            .resources
            .as_ref()
            .and_then(|resources| resources.devices.device_ordinals().iter().copied().min())
            .or_else(|| self.inner.fallback_device_ledger.as_ref().map(DeviceLedger::device_ordinal))
    }

    pub(super) fn max_active_attempt_limit(&self) -> usize {
        self.inner
            .resources
            .as_ref()
            .and_then(|resources| {
                resources.devices.contexts().map(|context| context.ledger().profile().max_active_attempts).max()
            })
            .or_else(|| self.inner.fallback_device_ledger.as_ref().map(|ledger| ledger.profile().max_active_attempts))
            .unwrap_or(0)
    }

    /// Returns true when both handles point at the same service instance.
    pub fn same_service(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Closes every scheduler ledger for an injected unknown-isolation fault.
    /// This test hook exercises the same controller transition as production
    /// cleanup; it does not introduce a second health latch.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn force_backend_unhealthy_for_test(&self, reason: &str) {
        self.inner.close_unknown_isolation_admission(reason);
    }

    /// Pauses a test-only unknown-isolation closure after its first ledger
    /// closes, while the fleet admission gate remains held.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn pause_unknown_isolation_closure_after_first_ledger_for_test(
        &self,
    ) -> UnknownIsolationClosurePauseHandle {
        let (entered_sender, entered_receiver) = std::sync::mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
        let pause =
            Arc::new(UnknownIsolationClosurePause { entered: entered_sender, release: Mutex::new(release_receiver) });
        let mut armed =
            self.inner.unknown_isolation_closure_pause.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(armed.is_none(), "only one unknown-isolation closure pause may be armed for a service");
        *armed = Some(pause);
        UnknownIsolationClosurePauseHandle {
            entered: Mutex::new(entered_receiver),
            release: Mutex::new(Some(release_sender)),
        }
    }

    /// Injects one failure after an exhausted waiter's old protection has
    /// ended during health-driven retargeting.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn fail_next_protection_retarget_after_release_for_test(&self) {
        self.inner.protection_retarget_failure_after_release.store(true, Ordering::Release);
    }

    /// Installs a fault-injection probe over the admission transaction.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn set_admission_transaction_probe(
        &self,
        probe: transaction::AdmissionTransactionProbe,
    ) {
        *self.inner.admission_transaction_probe.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(Arc::new(probe));
    }
}
