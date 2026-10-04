//! GPU-free admission state: the queue, the fleet gate, per-device ledgers,
//! and native-fault probes.

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, Condvar, Mutex, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use query_planning::{
    admission::ExecutionCapability,
    error::{QueryError, QueryErrorCode, Result},
};
use tokio::sync::Notify;

use super::{admission::AdmissionState, config::QueueBounds};

mod placement;
mod queue;
mod reporting;
mod shutdown;
mod submission;

pub(in crate::runtime::query_service) use placement::required_capabilities_for_placement;
pub(in crate::runtime::query_service) use queue::query_service_shutdown_timeout;
#[cfg(test)]
pub(in crate::runtime::query_service) use submission::PendingSubmission;

use crate::runtime::{DeviceLedger, gpu::LedgerTransitionKind};

/// Admission scheduling state shared by the query service, its waiters, and
/// the lifecycle tickets they hold. It owns no CUDA object.
#[derive(Debug)]
pub(in crate::runtime::query_service) struct AdmissionController {
    pub(in crate::runtime::query_service) queue_bounds: QueueBounds,
    pub(in crate::runtime::query_service) devices: AdmissionDevices,
    pub(in crate::runtime::query_service) admission_open: AtomicBool,
    /// Serializes every ledger admission transaction with a fleet-wide
    /// unknown-isolation closure. The queue mutex orders controller state;
    /// this gate additionally makes the multi-ledger health transition one
    /// atomic scheduler boundary.
    pub(in crate::runtime::query_service) fleet_admission: Mutex<()>,
    pub(in crate::runtime::query_service) state: Mutex<AdmissionState>,
    pub(in crate::runtime::query_service) admission_available: Condvar,
    pub(in crate::runtime::query_service) admission_notify: Notify,
    /// Advanced by each ledger transition that can make a queued request
    /// impossible (an admission close or a frozen-resident publish); a request
    /// rechecks device health and resident capacity when the generation it
    /// was checked at is older.
    pub(in crate::runtime::query_service) queue_revalidation_generation: AtomicU64,
    pub(in crate::runtime::query_service) next_admission_request_id: AtomicU64,
    /// Monotonic FIFO submission identity, independent of eventual admission order.
    pub(in crate::runtime::query_service) next_submission_seq: AtomicU64,
    pub(in crate::runtime::query_service) next_admission_sequence: AtomicU64,
    #[cfg(test)]
    pub(in crate::runtime::query_service) shutdown_waiters: AtomicUsize,
}

impl AdmissionController {
    /// Creates the controller and registers one admission observer per
    /// ledger.
    ///
    /// Grant releases wake the queue; the admission invariant forbids grant
    /// drops while the admission mutex is held. An admission close or a
    /// frozen-resident publish can happen under this mutex (placement and the
    /// submission sweep fold native faults while holding it), so the observer
    /// never takes it: it only advances the revalidation generation and wakes
    /// the queue, and each queued request rechecks its device state on its
    /// next step.
    pub(in crate::runtime::query_service) fn new(
        queue_bounds: QueueBounds,
        devices: AdmissionDevices,
    ) -> Result<Arc<Self>> {
        let controller = Arc::new(Self {
            queue_bounds,
            devices,
            admission_open: AtomicBool::new(true),
            fleet_admission: Mutex::new(()),
            state: Mutex::new(AdmissionState::default()),
            admission_available: Condvar::new(),
            admission_notify: Notify::new(),
            queue_revalidation_generation: AtomicU64::new(0),
            next_admission_request_id: AtomicU64::new(1),
            next_submission_seq: AtomicU64::new(1),
            next_admission_sequence: AtomicU64::new(1),
            #[cfg(test)]
            shutdown_waiters: AtomicUsize::new(0),
        });
        for ledger in controller.devices.ledgers() {
            let weak = Arc::downgrade(&controller);
            ledger.register_admission_observer(Arc::new(move |transition| {
                if let Some(controller) = weak.upgrade() {
                    if transition.kind != LedgerTransitionKind::GrantReleased {
                        controller.queue_revalidation_generation.fetch_add(1, Ordering::AcqRel);
                    }
                    controller.notify_admission_available();
                }
            }))?;
        }
        Ok(controller)
    }

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
        let _fleet_admission = self.fleet_admission.lock().unwrap_or_else(PoisonError::into_inner);
        let detail = detail.as_ref();
        for ledger in self.devices.ledgers() {
            if let Err(error) = ledger.close_admission_for_unknown_isolation(detail) {
                eprintln!(
                    "[query-engine] warning: failed to close device admission after unknown-isolation failure: {error}"
                );
            }
        }
        self.notify_admission_available();
    }
}

/// Closure reason once cuDF records a persistent fault on the device, or
/// `None` while it is healthy.
type NativeFaultProbe = Box<dyn Fn() -> Option<String> + Send + Sync>;

/// One device the admission scheduler places on.
///
/// The probe captures only the device identity, never a `DeviceContext`, so
/// the scheduler cannot keep stream pools or memory managers alive past the
/// service resources' teardown order.
pub(in crate::runtime::query_service) struct AdmissionDevice {
    ledger: Arc<DeviceLedger>,
    native_fault: NativeFaultProbe,
}

impl AdmissionDevice {
    pub(in crate::runtime::query_service) fn new(
        ledger: Arc<DeviceLedger>,
        native_fault: impl Fn() -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self { ledger, native_fault: Box::new(native_fault) }
    }
}

/// The scheduler's devices keyed by CUDA ordinal. Each ledger is the device's
/// only admission-health authority.
pub(in crate::runtime::query_service) struct AdmissionDevices {
    devices: BTreeMap<usize, AdmissionDevice>,
}

impl fmt::Debug for AdmissionDevices {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AdmissionDevices").field("ordinals", &self.devices.keys()).finish()
    }
}

impl AdmissionDevices {
    pub(in crate::runtime::query_service) fn new(devices: impl IntoIterator<Item = AdmissionDevice>) -> Self {
        Self { devices: devices.into_iter().map(|device| (device.ledger.device_ordinal(), device)).collect() }
    }

    /// The ledger of `ordinal`, with the device registry's unknown-ordinal
    /// error shape.
    pub(in crate::runtime::query_service) fn ledger(&self, ordinal: usize) -> Result<&Arc<DeviceLedger>> {
        self.device(ordinal).map(|device| &device.ledger)
    }

    fn device(&self, ordinal: usize) -> Result<&AdmissionDevice> {
        self.devices.get(&ordinal).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_registry.context",
                "CUDA device is not installed in the device registry",
            )
            .with_fact("device_ordinal", ordinal)
        })
    }

    /// Ledgers in ordinal order.
    pub(in crate::runtime::query_service) fn ledgers(&self) -> impl Iterator<Item = &Arc<DeviceLedger>> {
        self.devices.values().map(|device| &device.ledger)
    }

    pub(in crate::runtime::query_service) fn len(&self) -> usize {
        self.devices.len()
    }

    /// Whether the immutable profile of `ordinal` provides every capability.
    pub(in crate::runtime::query_service) fn supports_capabilities(
        &self,
        ordinal: usize,
        required_capabilities: &[ExecutionCapability],
    ) -> bool {
        self.devices
            .get(&ordinal)
            .is_some_and(|device| device.ledger.profile().device_capabilities.supports_all(required_capabilities))
    }

    /// Folds a persistent native fault on `ordinal` into its ledger; returns
    /// whether this call closed admission. The closing ledger notifies the
    /// service through its admission observer.
    pub(in crate::runtime::query_service) fn synchronize_native_health(&self, ordinal: usize) -> Result<bool> {
        let device = self.device(ordinal)?;
        (device.native_fault)().map_or(Ok(false), |reason| device.ledger.close_admission_if_open(reason))
    }

    /// Folds persistent native faults on every device into their ledgers.
    pub(in crate::runtime::query_service) fn reconcile_native_health(&self) -> Result<()> {
        for ordinal in self.devices.keys() {
            self.synchronize_native_health(*ordinal)?;
        }
        Ok(())
    }
}
