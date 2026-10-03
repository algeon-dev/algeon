//! Process-lifetime CUDA device health and fail-closed teardown support.
//!
//! Device health is shared by every cudf/cugraph owner in the process. Once a
//! device-bound CUDA failure makes safe future entry uncertain, the device is
//! permanently unhealthy and normal work is rejected. Teardown may still
//! attempt to enter the immutable owner device, but native destruction is
//! suppressed when that entry cannot be proven.

use std::{
    collections::HashMap,
    fmt,
    io::Write,
    ptr,
    sync::{
        Arc, Mutex, OnceLock, PoisonError, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use super::DeviceId;
use crate::core::error::{Error, ErrorCode, Result, check_status_in_without_device_report};
#[cfg(any(test, feature = "test-hooks"))]
use crate::testing::device_safety::injected_health_registry;

/// Process-lifetime usability of one CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceHealthStatus {
    /// New device-bound work may be admitted.
    Healthy,
    /// A device-bound fault occurred; new work is permanently rejected.
    Unhealthy,
}

/// Machine-readable reason a CUDA device was first marked unhealthy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceFaultKind {
    /// The calling thread's current CUDA device could not be queried.
    CurrentDeviceQueryFailed,
    /// The calling thread could not enter the immutable owner device.
    OwnerDeviceEntryFailed,
    /// The calling thread's prior ambient CUDA device could not be restored.
    AmbientDeviceRestoreFailed,
    /// Device-bound FFI reported a fatal CUDA status.
    FatalDeviceOperation,
    /// A device-bound destructor was suppressed because safe teardown could
    /// not be proven.
    TeardownQuarantined,
}

/// First fault retained for one unhealthy CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceFault {
    kind: DeviceFaultKind,
    operation: &'static str,
    native_status: Option<DeviceFaultNativeStatus>,
}

/// Native status retained with a device fault, including its ABI family.
///
/// cuDF status codes and CUDA runtime status codes have distinct native types.
/// Retaining the family prevents lossy signedness conversions when a fault is
/// reported across the runtime boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceFaultNativeStatus {
    /// Status produced by a cuDF FFI call.
    Cudf(cudf_sys::cudf_status_t),
    /// Status produced by the CUDA runtime.
    Cuda(i32),
}

impl DeviceFaultNativeStatus {
    /// Native cuDF status used when a cuDF call reports a fatal CUDA failure.
    pub const CUDF_FATAL_CUDA: Self = Self::Cudf(cudf_sys::CUDF_ERROR_FATAL_CUDA);
}

impl DeviceFault {
    /// Fault classification.
    pub fn kind(self) -> DeviceFaultKind {
        self.kind
    }

    /// cudf/cugraph boundary where the fault was observed.
    pub fn operation(self) -> &'static str {
        self.operation
    }

    /// Native status and its ABI family when the fault originated from a
    /// status-returning FFI call.
    pub fn native_status(self) -> Option<DeviceFaultNativeStatus> {
        self.native_status
    }
}

/// Structured process-lifetime health snapshot for one CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceHealthSnapshot {
    device: DeviceId,
    status: DeviceHealthStatus,
    first_fault: Option<DeviceFault>,
    fault_count: u64,
    rejected_operation_count: u64,
    quarantined_teardown_count: u64,
}

impl DeviceHealthSnapshot {
    /// Device described by this snapshot.
    pub fn device(self) -> DeviceId {
        self.device
    }

    /// Current process-lifetime admission status.
    pub fn status(self) -> DeviceHealthStatus {
        self.status
    }

    /// First fault that permanently closed admission, if any.
    pub fn first_fault(self) -> Option<DeviceFault> {
        self.first_fault
    }

    /// Saturating count of recorded device faults.
    pub fn fault_count(self) -> u64 {
        self.fault_count
    }

    /// Saturating count of normal operations rejected after a fault.
    pub fn rejected_operation_count(self) -> u64 {
        self.rejected_operation_count
    }

    /// Saturating count of native resources intentionally left allocated
    /// because teardown could not complete safely.
    pub fn quarantined_teardown_count(self) -> u64 {
        self.quarantined_teardown_count
    }
}

type DeviceFaultObserver = Arc<dyn Fn(DeviceHealthSnapshot) + Send + Sync>;

pub(crate) struct DeviceHealthState {
    unhealthy: AtomicBool,
    first_fault: OnceLock<DeviceFault>,
    fault_count: AtomicU64,
    rejected_operation_count: AtomicU64,
    quarantined_teardown_count: AtomicU64,
    fault_observers: Mutex<Vec<(u64, DeviceFaultObserver)>>,
    next_fault_observer: AtomicU64,
}

impl DeviceHealthState {
    fn new() -> Self {
        Self {
            unhealthy: AtomicBool::new(false),
            first_fault: OnceLock::new(),
            fault_count: AtomicU64::new(0),
            rejected_operation_count: AtomicU64::new(0),
            quarantined_teardown_count: AtomicU64::new(0),
            fault_observers: Mutex::new(Vec::new()),
            next_fault_observer: AtomicU64::new(0),
        }
    }

    fn fault_observers(&self) -> Vec<DeviceFaultObserver> {
        let observers = self.fault_observers.lock().unwrap_or_else(PoisonError::into_inner);
        observers.iter().map(|(_, observer)| Arc::clone(observer)).collect()
    }

    pub(crate) fn snapshot(&self, device: DeviceId) -> DeviceHealthSnapshot {
        DeviceHealthSnapshot {
            device,
            status: if self.unhealthy.load(Ordering::Acquire) {
                DeviceHealthStatus::Unhealthy
            } else {
                DeviceHealthStatus::Healthy
            },
            first_fault: self.first_fault.get().copied(),
            fault_count: self.fault_count.load(Ordering::Relaxed),
            rejected_operation_count: self.rejected_operation_count.load(Ordering::Relaxed),
            quarantined_teardown_count: self.quarantined_teardown_count.load(Ordering::Relaxed),
        }
    }
}

#[derive(Default)]
pub(crate) struct DeviceHealthRegistry {
    states: Mutex<HashMap<i32, Arc<DeviceHealthState>>>,
}

impl DeviceHealthRegistry {
    pub(crate) fn state(&self, device: DeviceId) -> Arc<DeviceHealthState> {
        let mut states = self.states.lock().expect("device health registry mutex poisoned");
        Arc::clone(states.entry(device.ordinal_raw()).or_insert_with(|| Arc::new(DeviceHealthState::new())))
    }
}

fn global_registry() -> &'static DeviceHealthRegistry {
    static REGISTRY: OnceLock<DeviceHealthRegistry> = OnceLock::new();
    REGISTRY.get_or_init(DeviceHealthRegistry::default)
}

fn with_registry<R>(f: impl FnOnce(&DeviceHealthRegistry) -> R) -> R {
    #[cfg(any(test, feature = "test-hooks"))]
    if let Some(registry) = injected_health_registry() {
        return f(&registry);
    }
    f(global_registry())
}

fn saturating_increment(counter: &AtomicU64) -> u64 {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        if current == u64::MAX {
            return u64::MAX;
        }
        match counter.compare_exchange_weak(current, current + 1, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return current + 1,
            Err(observed) => current = observed,
        }
    }
}

fn should_log(count: u64) -> bool {
    count == 1 || count.is_power_of_two()
}

fn write_diagnostic(message: fmt::Arguments<'_>) {
    let _ = writeln!(std::io::stderr().lock(), "{message}");
}

/// Read the process-lifetime health of `device`.
pub fn device_health_snapshot(device: DeviceId) -> DeviceHealthSnapshot {
    with_registry(|registry| registry.state(device).snapshot(device))
}

/// A registration of [`observe_device_faults`]; dropping it removes the
/// observer.
#[must_use = "dropping the registration stops the observation"]
pub struct DeviceFaultObservation {
    state: Weak<DeviceHealthState>,
    id: u64,
}

impl fmt::Debug for DeviceFaultObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("DeviceFaultObservation").field("id", &self.id).finish_non_exhaustive()
    }
}

impl Drop for DeviceFaultObservation {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            state.fault_observers.lock().unwrap_or_else(PoisonError::into_inner).retain(|(id, _)| *id != self.id);
        }
    }
}

/// Runs `observer` with the device's health each time a fault is reported
/// for `device`, until the returned registration drops. The observer runs on
/// the reporting thread after the fault is recorded, outside every health
/// lock. A fault reported before registration is not replayed; read
/// [`device_health_snapshot`] after registering to cover it.
pub fn observe_device_faults(
    device: DeviceId,
    observer: impl Fn(DeviceHealthSnapshot) + Send + Sync + 'static,
) -> DeviceFaultObservation {
    with_registry(|registry| {
        let state = registry.state(device);
        let id = state.next_fault_observer.fetch_add(1, Ordering::Relaxed);
        state.fault_observers.lock().unwrap_or_else(PoisonError::into_inner).push((id, Arc::new(observer)));
        DeviceFaultObservation { state: Arc::downgrade(&state), id }
    })
}

pub(crate) fn ensure_healthy(device: DeviceId, operation: &'static str) -> Result<()> {
    with_registry(|registry| {
        let state = registry.state(device);
        if !state.unhealthy.load(Ordering::Acquire) {
            return Ok(());
        }

        let rejected = saturating_increment(&state.rejected_operation_count);
        if should_log(rejected) {
            write_diagnostic(format_args!(
                "[cudf] device_work_rejected device={} operation={operation} rejected_count={rejected}",
                device.ordinal_raw()
            ));
        }
        let snapshot = state.snapshot(device);
        let mut error = Error::new_with(
            ErrorCode::FatalCuda,
            operation,
            "CUDA device is unhealthy; process-lifetime safety policy rejects new work",
        )
        .with_detail("device", device.ordinal().to_string())
        .with_detail("device_health", "unhealthy")
        .with_detail("reason", "prior_device_fault")
        .with_detail("remedy", "stop admitting work on this device and restart the process after GPU recovery");
        if let Some(first) = snapshot.first_fault() {
            error = error
                .with_detail("first_fault_kind", format!("{:?}", first.kind()))
                .with_detail("first_fault_operation", first.operation());
        }
        Err(error)
    })
}

pub(crate) fn record_fault(
    device: DeviceId,
    kind: DeviceFaultKind,
    operation: &'static str,
    native_status: Option<cudf_sys::cudf_status_t>,
) {
    report_device_fault(device, kind, operation, native_status.map(DeviceFaultNativeStatus::Cudf));
}

/// Report a device-bound fault observed by an authorized sibling runtime.
///
/// This is intended for native libraries that execute under
/// [`DeviceEntryGuard`](super::DeviceEntryGuard) but receive their fatal status
/// through a non-cudf FFI boundary. Reporting is process-lifetime and
/// immediately closes normal admission for `device`.
pub fn report_device_fault(
    device: DeviceId,
    kind: DeviceFaultKind,
    operation: &'static str,
    native_status: Option<DeviceFaultNativeStatus>,
) {
    let (snapshot, observers) = with_registry(|registry| {
        let state = registry.state(device);
        let fault = DeviceFault { kind, operation, native_status };
        let _ = state.first_fault.set(fault);
        state.unhealthy.store(true, Ordering::Release);
        let count = saturating_increment(&state.fault_count);
        if should_log(count) {
            write_diagnostic(format_args!(
                "[cudf] device_fault device={} kind={kind:?} operation={operation} status={native_status:?} fault_count={count}",
                device.ordinal_raw()
            ));
        }
        (state.snapshot(device), state.fault_observers())
    });
    for observer in observers {
        observer(snapshot);
    }
}

pub(crate) fn check_device_status_in(
    status: cudf_sys::cudf_status_t,
    device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    if status == cudf_sys::CUDF_ERROR_FATAL_CUDA {
        record_fault(device, DeviceFaultKind::FatalDeviceOperation, operation, Some(status));
    }
    check_status_in_without_device_report(status, operation)
}

/// Consume one device-owned pointer and invoke its destructor only after the
/// current host thread has proven entry to the immutable owner device.
///
/// When the thread holds another device's entry, the teardown entry suspends
/// it instead of quarantining the teardown device. On entry failure ownership
/// is still consumed exactly once, native FFI is not called, and the
/// allocation is quarantined for process lifetime.
pub fn drop_device_resource<T>(
    raw: &mut *mut T,
    device: DeviceId,
    operation: &'static str,
    destroy: impl FnOnce(*mut T),
) {
    let owned = std::mem::replace(raw, ptr::null_mut());
    if owned.is_null() {
        return;
    }

    match super::device_guard::DeviceEntryGuard::enter_for_teardown(device, operation) {
        Ok(_entry) => destroy(owned),
        Err(error) => quarantine_device_resource(device, operation, &error),
    }
}

/// Record a native resource intentionally leaked because its device-bound
/// teardown could not be made safe.
pub fn quarantine_device_resource(device: DeviceId, operation: &'static str, reason: &dyn fmt::Display) {
    with_registry(|registry| {
        let state = registry.state(device);
        let count = saturating_increment(&state.quarantined_teardown_count);
        let quarantine_is_first_fault = if state.first_fault.get().is_none() {
            state
                .first_fault
                .set(DeviceFault { kind: DeviceFaultKind::TeardownQuarantined, operation, native_status: None })
                .is_ok()
        } else {
            false
        };
        state.unhealthy.store(true, Ordering::Release);
        if quarantine_is_first_fault {
            let _ = saturating_increment(&state.fault_count);
        }
        if should_log(count) {
            write_diagnostic(format_args!(
                "[cudf] device_teardown_quarantined device={} operation={operation} quarantined_count={count} reason={reason}",
                device.ordinal_raw()
            ));
        }
    });
}

pub(crate) fn record_teardown_failure(device: DeviceId, operation: &'static str, reason: &dyn fmt::Display) {
    with_registry(|registry| {
        let state = registry.state(device);
        let count = saturating_increment(&state.quarantined_teardown_count);
        if should_log(count) {
            write_diagnostic(format_args!(
                "[cudf] device_teardown_failed device={} operation={operation} quarantined_count={count} reason={reason}",
                device.ordinal_raw()
            ));
        }
    });
}

/// Write a failed cross-stream completion fence and the host-synchronization
/// result that replaced it to the device diagnostic stream.
///
/// The fence's device-entry and fatal CUDA failures already record a device
/// fault at their source, so this only reports the outcome.
pub(crate) fn report_completion_fence_fallback(
    device: DeviceId,
    operation: &'static str,
    fence_error: &dyn fmt::Display,
    fallback: &Result<()>,
) {
    match fallback {
        Ok(()) => write_diagnostic(format_args!(
            "[cudf] completion_fence_host_fallback device={} operation={operation} reason={fence_error}",
            device.ordinal_raw()
        )),
        Err(sync_error) => write_diagnostic(format_args!(
            "[cudf] completion_fence_failed device={} operation={operation} reason={fence_error} host_fallback_error={sync_error} consequence=cross_stream_inputs_may_be_reused_before_consumer_completes",
            device.ordinal_raw()
        )),
    }
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn new_test_registry() -> Arc<DeviceHealthRegistry> {
    Arc::new(DeviceHealthRegistry::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        runtime::device_guard::{DeviceEntryGuard, current_entered_device},
        testing::device_safety::{DeviceEntryTestCall, DeviceSafetyTestScope, injected_device_id},
    };

    #[test]
    fn counter_increment_saturates() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(saturating_increment(&counter), u64::MAX);
        assert_eq!(saturating_increment(&counter), u64::MAX);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn diagnostics_are_first_and_power_of_two_only() {
        let emitted: Vec<_> = (1..=17).filter(|count| should_log(*count)).collect();
        assert_eq!(emitted, vec![1, 2, 4, 8, 16]);
    }

    #[test]
    fn first_fault_is_stable_and_later_work_is_rejected() {
        let scope = DeviceSafetyTestScope::install([]);
        let device = DeviceId::DEFAULT;
        record_fault(
            device,
            DeviceFaultKind::OwnerDeviceEntryFailed,
            "first_fault",
            Some(cudf_sys::CUDF_ERROR_FATAL_CUDA),
        );
        record_fault(
            device,
            DeviceFaultKind::FatalDeviceOperation,
            "later_fault",
            Some(cudf_sys::CUDF_ERROR_FATAL_CUDA),
        );

        let snapshot = scope.health_snapshot(device);
        assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.fault_count(), 2);
        assert_eq!(snapshot.first_fault().expect("first fault").operation(), "first_fault");
        let error = ensure_healthy(device, "new_work").expect_err("unhealthy device must reject new work");
        assert_eq!(error.code(), ErrorCode::FatalCuda);
        assert_eq!(scope.health_snapshot(device).rejected_operation_count(), 1);
    }

    #[test]
    fn cudf_fault_status_retains_its_unsigned_ffi_value() {
        let scope = DeviceSafetyTestScope::install([]);
        let device = DeviceId::DEFAULT;
        let status = u32::MAX;

        report_device_fault(
            device,
            DeviceFaultKind::FatalDeviceOperation,
            "unsigned_cudf_status",
            Some(DeviceFaultNativeStatus::Cudf(status)),
        );

        let first = scope.health_snapshot(device).first_fault().expect("recorded fault");
        assert_eq!(first.native_status(), Some(DeviceFaultNativeStatus::Cudf(status)));
    }

    /// Dropping another device's resource while this thread holds an entry
    /// suspends for one teardown retry instead of quarantining the teardown
    /// device. Without the retry the nested entry fails, the destructor never
    /// runs, and the teardown device is marked unhealthy for the process.
    #[test]
    fn cross_device_teardown_suspends_instead_of_quarantining() {
        let scope = DeviceSafetyTestScope::install([
            DeviceEntryTestCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 1 },
            DeviceEntryTestCall::SetDevice { expected_target: 0, status: cudf_sys::CUDF_SUCCESS },
            DeviceEntryTestCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 0 },
            DeviceEntryTestCall::SetDevice { expected_target: 1, status: cudf_sys::CUDF_SUCCESS },
            DeviceEntryTestCall::SetDevice { expected_target: 0, status: cudf_sys::CUDF_SUCCESS },
            DeviceEntryTestCall::SetDevice { expected_target: 1, status: cudf_sys::CUDF_SUCCESS },
        ]);
        let active = injected_device_id(0).expect("active device");
        let teardown = injected_device_id(1).expect("teardown device");

        let _entry = DeviceEntryGuard::enter(active, "cross_device_teardown").expect("active entry");
        let mut destroyed = false;
        let mut raw = ptr::dangling_mut::<u8>();
        drop_device_resource(&mut raw, teardown, "cross_device_teardown", |owned| {
            assert_eq!(owned, ptr::dangling_mut::<u8>());
            destroyed = true;
        });

        assert!(raw.is_null(), "teardown consumes ownership exactly once");
        assert!(destroyed, "the destructor must run instead of quarantining");
        assert_eq!(current_entered_device(), Some(active), "the suspended entry resumes");
        assert_eq!(scope.health_snapshot(teardown).status(), DeviceHealthStatus::Healthy);
        assert_eq!(scope.health_snapshot(teardown).fault_count(), 0);
        assert_eq!(scope.health_snapshot(active).fault_count(), 0);
        drop(_entry);
        assert_eq!(scope.remaining_call_count(), 0);
    }

    #[test]
    fn nonfatal_native_teardown_failure_is_observable_without_poisoning_admission() {
        let scope = DeviceSafetyTestScope::install([]);
        let device = DeviceId::DEFAULT;
        record_teardown_failure(device, "nonfatal_destroy", &"invalid native teardown state");

        let snapshot = scope.health_snapshot(device);
        assert_eq!(snapshot.status(), DeviceHealthStatus::Healthy);
        assert_eq!(snapshot.fault_count(), 0);
        assert_eq!(snapshot.quarantined_teardown_count(), 1);
        ensure_healthy(device, "later_work").expect("nonfatal teardown rejection must not poison device admission");
    }
}
