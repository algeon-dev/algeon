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
    DeviceQuarantineInjection, QueryActivityLease, QueryAdmissionRequest, QueryServiceAdmissionSnapshot,
    QueryServiceDeviceAdmissionSnapshot,
};
pub(in crate::runtime::query_service::service) use queue::query_service_shutdown_timeout;
pub(in crate::runtime::query_service) use requests::PlacedAdmissionCommit;

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

/// Test-side half of a one-shot pause rendezvous.
///
/// Dropping the handle releases a paused worker so a failing assertion cannot
/// strand a test thread. Optional cleanup runs after that release, matching
/// the registry-backed pause handles' previous teardown order.
#[cfg(test)]
pub(in crate::runtime::query_service) struct TestPauseHandle {
    name: &'static str,
    entered: Mutex<std::sync::mpsc::Receiver<()>>,
    release: Mutex<Option<std::sync::mpsc::SyncSender<()>>>,
    drop_cleanup: Mutex<Option<Box<dyn FnOnce() + Send>>>,
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
        entered: std::sync::mpsc::Receiver<()>,
        release: std::sync::mpsc::SyncSender<()>,
    ) -> Self {
        Self { name, entered: Mutex::new(entered), release: Mutex::new(Some(release)), drop_cleanup: Mutex::new(None) }
    }

    pub(in crate::runtime::query_service) fn with_drop_cleanup(
        mut self,
        cleanup: impl FnOnce() + Send + 'static,
    ) -> Self {
        *self.drop_cleanup.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Box::new(cleanup));
        self
    }

    pub(in crate::runtime::query_service) fn wait_until_entered(&self, timeout: Duration) {
        self.entered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv_timeout(timeout)
            .unwrap_or_else(|error| panic!("{} did not enter before the deadline: {error}", self.name));
    }

    pub(in crate::runtime::query_service) fn entered_within(&self, timeout: Duration) -> bool {
        match self.entered.lock().unwrap_or_else(std::sync::PoisonError::into_inner).recv_timeout(timeout) {
            Ok(()) => true,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => false,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                panic!("{} pause sender disconnected", self.name)
            },
        }
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
        if let Some(cleanup) = self.drop_cleanup.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            cleanup();
        }
    }
}

#[cfg(test)]
mod test_pause_handle_tests {
    use std::{sync::mpsc, time::Duration};

    use super::TestPauseHandle;

    #[test]
    fn drop_releases_the_paused_worker_before_running_cleanup() {
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (release_sender, release_receiver) = mpsc::sync_channel(1);
        let (worker_released_sender, worker_released_receiver) = mpsc::sync_channel(1);
        let (cleanup_sender, cleanup_receiver) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            entered_sender.send(()).expect("pause handle remains live until the worker enters");
            release_receiver.recv().expect("pause handle releases the worker during teardown");
            worker_released_sender.send(()).expect("drop cleanup remains live until the worker is released");
        });
        let pause = TestPauseHandle::new("test pause", entered_receiver, release_sender).with_drop_cleanup(move || {
            worker_released_receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("pause handle releases the worker before running cleanup");
            cleanup_sender.send(()).expect("test observes pause cleanup");
        });

        pause.wait_until_entered(Duration::from_secs(2));
        drop(pause);

        cleanup_receiver.recv_timeout(Duration::from_secs(2)).expect("drop runs pause cleanup");
        worker.join().expect("paused worker does not panic");
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

    /// Performs an explicit, controlled device-local quarantine transition.
    ///
    /// This is intentionally an engine capability rather than a backend
    /// configuration mode. It owns the same ledger health state consulted by
    /// placement, takes the fleet admission gate used by placement, and leaves
    /// native cuDF health untouched. A caller can therefore validate fallback
    /// after a quarantine without claiming to have produced a real CUDA fault.
    pub(crate) fn inject_device_quarantine(&self, device_ordinal: usize) -> Result<DeviceQuarantineInjection> {
        const REASON: &str = "controlled validation device quarantine";

        let injection = {
            let _fleet_admission = self.inner.fleet_admission.lock().map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.fault_injection",
                    "fleet admission gate was poisoned",
                )
                .with_fact("reason", "fleet_admission_gate_poisoned")
            })?;
            let resources = self.inner.resources.as_ref().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.fault_injection",
                    "controlled device quarantine requires an installed GPU backend",
                )
                .with_fact("reason", "gpu_backend_required")
            })?;
            let ledger = resources.devices.context_by_ordinal(device_ordinal)?.ledger();
            let newly_quarantined = ledger.close_admission_if_open(REASON)?;
            let crate::runtime::gpu::DeviceAdmissionHealth::Closed { generation, .. } = ledger.health()? else {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.fault_injection",
                    "device ledger remained open after controlled quarantine",
                )
                .with_fact("reason", "quarantine_transition_not_observed")
                .with_fact("device_ordinal", device_ordinal));
            };
            DeviceQuarantineInjection::new(device_ordinal, generation, newly_quarantined)
        };
        self.inner.notify_admission_available();
        Ok(injection)
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
    ) -> TestPauseHandle {
        let (entered_sender, entered_receiver) = std::sync::mpsc::sync_channel(1);
        let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
        let pause =
            Arc::new(UnknownIsolationClosurePause { entered: entered_sender, release: Mutex::new(release_receiver) });
        let mut armed =
            self.inner.unknown_isolation_closure_pause.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(armed.is_none(), "only one unknown-isolation closure pause may be armed for a service");
        *armed = Some(pause);
        TestPauseHandle::new("unknown-isolation closure", entered_receiver, release_sender)
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
