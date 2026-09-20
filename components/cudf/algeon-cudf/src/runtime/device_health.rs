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
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use super::{
    DeviceId,
    device_guard::{SuspendedDeviceEntry, current_entered_device},
};
use crate::{
    core::error::{Error, ErrorCode, Result, check_status_in_without_device_report},
    testing::device_safety::injected_health_registry,
};

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

pub(crate) struct DeviceHealthState {
    unhealthy: AtomicBool,
    first_fault: OnceLock<DeviceFault>,
    fault_count: AtomicU64,
    rejected_operation_count: AtomicU64,
    quarantined_teardown_count: AtomicU64,
}

impl DeviceHealthState {
    fn new() -> Self {
        Self {
            unhealthy: AtomicBool::new(false),
            first_fault: OnceLock::new(),
            fault_count: AtomicU64::new(0),
            rejected_operation_count: AtomicU64::new(0),
            quarantined_teardown_count: AtomicU64::new(0),
        }
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
    unknown_device_quarantined_count: AtomicU64,
}

impl DeviceHealthRegistry {
    pub(crate) fn state(&self, device: DeviceId) -> Arc<DeviceHealthState> {
        let mut states = self.states.lock().expect("device health registry mutex poisoned");
        Arc::clone(states.entry(device.ordinal_raw()).or_insert_with(|| Arc::new(DeviceHealthState::new())))
    }

    pub(crate) fn unknown_device_quarantined_count(&self) -> u64 {
        self.unknown_device_quarantined_count.load(Ordering::Relaxed)
    }
}

fn global_registry() -> &'static DeviceHealthRegistry {
    static REGISTRY: OnceLock<DeviceHealthRegistry> = OnceLock::new();
    REGISTRY.get_or_init(DeviceHealthRegistry::default)
}

fn with_registry<R>(f: impl FnOnce(&DeviceHealthRegistry) -> R) -> R {
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
    with_registry(|registry| {
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
    });
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

/// Teardown retries after suspending the thread's active device entry.
///
/// The suspend drains the thread's device stack, so one retry always reaches
/// an empty stack and either enters the teardown device or proves a genuine
/// entry failure worth quarantining.
const CROSS_DEVICE_TEARDOWN_RETRIES: u32 = 1;

/// Consume one device-owned pointer and invoke its destructor only after the
/// current host thread has proven entry to the immutable owner device.
///
/// When the thread holds another device's entry, that entry suspends for one
/// teardown retry instead of quarantining the teardown device. On entry
/// failure ownership is still consumed exactly once, native FFI is not
/// called, and the allocation is quarantined for process lifetime.
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

    let mut suspended: Option<SuspendedDeviceEntry> = None;
    for _ in 0..=CROSS_DEVICE_TEARDOWN_RETRIES {
        match super::device_guard::DeviceEntryGuard::enter_for_teardown(device, operation) {
            // The entry guard drops before `suspended`, so its ambient
            // restore and pop happen before the suspended entries resume.
            Ok(_entry) => return destroy(owned),
            Err(error) => {
                let retryable = suspended.is_none() && current_entered_device().is_some_and(|active| active != device);
                if !retryable {
                    return quarantine_device_resource(device, operation, &error);
                }
                match super::device_guard::DeviceEntryGuard::suspend_active_device() {
                    Some(entry) => suspended = Some(entry),
                    None => return quarantine_device_resource(device, operation, &error),
                }
            },
        }
    }
    // Only reachable with a zero retry budget: fail closed like any other
    // entry failure.
    quarantine_device_resource(device, operation, &"cross-device teardown retries exhausted");
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

/// Consume an owner-unknown device pointer without crossing native teardown.
///
/// No device can be safely selected or poisoned when provenance is absent, so
/// this path records a process-lifetime quarantine and intentionally leaks the
/// now-unreachable allocation rather than guessing an owner device.
pub(crate) fn quarantine_unknown_device_resource(operation: &'static str, reason: &dyn fmt::Display) {
    with_registry(|registry| {
        let count = saturating_increment(&registry.unknown_device_quarantined_count);
        if should_log(count) {
            write_diagnostic(format_args!(
                "[cudf] unknown_device_teardown_quarantined operation={operation} quarantined_count={count} reason={reason}"
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
