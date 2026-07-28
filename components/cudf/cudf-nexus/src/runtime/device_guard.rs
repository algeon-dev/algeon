//! Host-thread CUDA device entry.
//!
//! A device entry temporarily selects an owner's immutable CUDA device on the
//! calling host thread and restores the thread's prior ambient device on drop.
//! It does not migrate a query or any GPU allocation between devices.

use std::{cell::RefCell, marker::PhantomData};

use super::{
    DeviceId,
    device_health::{self, DeviceFaultKind},
};
use crate::core::error::{self, Error, ErrorCode, Result};

/// Thread-scoped proof that the current host thread entered one owner device.
///
/// The guard is `!Send`/`!Sync` because CUDA's current device is host-thread
/// local. Dropping it restores the previously ambient device when a change was
/// required.
#[derive(Debug)]
pub struct DeviceEntryGuard {
    owner: DeviceId,
    _not_send_sync: PhantomData<*mut ()>,
}

#[derive(Debug, Clone, Copy)]
struct ThreadAuthorityPoison {
    device: DeviceId,
    operation: &'static str,
    native_status: cudf_sys::cudf_status_t,
}

#[derive(Debug, Default)]
struct ThreadDeviceAuthority {
    entered_devices: Vec<DeviceId>,
    ambient_device: Option<DeviceId>,
    poison: Option<ThreadAuthorityPoison>,
}

thread_local! {
    static THREAD_AUTHORITY: RefCell<ThreadDeviceAuthority> = const {
        RefCell::new(ThreadDeviceAuthority {
            entered_devices: Vec::new(),
            ambient_device: None,
            poison: None,
        })
    };
}

fn thread_authority_error(operation: &'static str, poison: ThreadAuthorityPoison) -> Error {
    Error::new_with(
        ErrorCode::FatalCuda,
        operation,
        "CUDA host-thread device authority was invalidated by a failed device transition",
    )
    .with_detail("reason", "thread_device_authority_invalidated")
    .with_detail("fault_device", poison.device.ordinal().to_string())
    .with_detail("fault_operation", poison.operation)
    .with_detail("native_status", poison.native_status.to_string())
    .with_detail("remedy", "stop using this host thread for CUDA work")
}

fn current_thread_poison() -> Option<ThreadAuthorityPoison> {
    THREAD_AUTHORITY.with(|authority| authority.borrow().poison)
}

fn poison_thread_authority(poison: ThreadAuthorityPoison) {
    THREAD_AUTHORITY.with(|authority| {
        let mut authority = authority.borrow_mut();
        authority.poison.get_or_insert(poison);
    });
}

fn translate_explicit_device_status(status: cudf_sys::cudf_status_t, operation: &'static str) -> Error {
    error::check_status_in_without_device_report(status, operation)
        .expect_err("non-success explicit device status must produce an error")
}

fn nested_cross_device_error(active: DeviceId, requested: DeviceId, operation: &'static str) -> Error {
    Error::new_with(
        ErrorCode::RuntimeBindingMismatch,
        operation,
        "nested CUDA device entry cannot change the active owner device",
    )
    .with_detail("reason", "nested_cross_device_entry")
    .with_detail("active_device", active.ordinal().to_string())
    .with_detail("requested_device", requested.ordinal().to_string())
}

impl DeviceEntryGuard {
    /// Enter `owner` for normal work, rejecting a process-lifetime unhealthy
    /// device before making any CUDA call.
    pub fn enter(owner: DeviceId, operation: &'static str) -> Result<Self> {
        Self::enter_impl(owner, operation, true)
    }

    pub(crate) fn new(owner: DeviceId) -> Result<Self> {
        Self::enter(owner, "runtime::device_entry")
    }

    /// Teardown bypasses the unhealthy admission check, but must still prove
    /// owner-device entry before a native destructor may run.
    pub(crate) fn enter_for_teardown(owner: DeviceId, operation: &'static str) -> Result<Self> {
        Self::enter_impl(owner, operation, false)
    }

    fn enter_impl(owner: DeviceId, operation: &'static str, require_healthy: bool) -> Result<Self> {
        if let Some(poison) = current_thread_poison() {
            return Err(thread_authority_error(operation, poison));
        }

        let active_owner = current_entered_device();
        if let Some(active_owner) = active_owner {
            if active_owner != owner {
                return Err(nested_cross_device_error(active_owner, owner, operation));
            }
            if require_healthy {
                device_health::ensure_healthy(owner, operation)?;
            }
            THREAD_AUTHORITY.with(|authority| authority.borrow_mut().entered_devices.push(owner));
            return Ok(Self { owner, _not_send_sync: PhantomData });
        }

        if require_healthy {
            device_health::ensure_healthy(owner, operation)?;
        }

        let mut current = 0;
        let get_status = crate::testing::device_safety::cuda_get_device(&mut current);
        if get_status != cudf_sys::CUDF_SUCCESS {
            device_health::record_fault(owner, DeviceFaultKind::CurrentDeviceQueryFailed, operation, Some(get_status));
            poison_thread_authority(ThreadAuthorityPoison { device: owner, operation, native_status: get_status });
            return Err(translate_explicit_device_status(get_status, operation));
        }
        let previous = DeviceId::from_validated_ordinal_raw(current);

        if previous != owner {
            let set_status = crate::testing::device_safety::cuda_set_device(owner.ordinal_raw());
            if set_status != cudf_sys::CUDF_SUCCESS {
                device_health::record_fault(
                    owner,
                    DeviceFaultKind::OwnerDeviceEntryFailed,
                    operation,
                    Some(set_status),
                );
                poison_thread_authority(ThreadAuthorityPoison { device: owner, operation, native_status: set_status });
                return Err(translate_explicit_device_status(set_status, operation));
            }
        }

        THREAD_AUTHORITY.with(|authority| {
            let mut authority = authority.borrow_mut();
            debug_assert!(authority.entered_devices.is_empty());
            debug_assert!(authority.ambient_device.is_none());
            authority.ambient_device = Some(previous);
            authority.entered_devices.push(owner);
        });
        Ok(Self { owner, _not_send_sync: PhantomData })
    }

    /// Device whose entry this guard proves.
    pub fn device(&self) -> DeviceId {
        self.owner
    }
}

impl Drop for DeviceEntryGuard {
    fn drop(&mut self) {
        let restore_device = THREAD_AUTHORITY.with(|authority| {
            let mut authority = authority.borrow_mut();
            let popped = authority.entered_devices.pop();
            debug_assert_eq!(popped, Some(self.owner), "device entries must reference one nested owner device");
            if authority.entered_devices.is_empty() {
                authority.ambient_device.take().filter(|ambient| *ambient != self.owner)
            } else {
                None
            }
        });

        if let Some(restore_device) = restore_device {
            let status = crate::testing::device_safety::cuda_set_device(restore_device.ordinal_raw());
            if status != cudf_sys::CUDF_SUCCESS {
                device_health::record_fault(
                    restore_device,
                    DeviceFaultKind::AmbientDeviceRestoreFailed,
                    "runtime::DeviceEntryGuard::drop",
                    Some(status),
                );
                poison_thread_authority(ThreadAuthorityPoison {
                    device: restore_device,
                    operation: "runtime::DeviceEntryGuard::drop",
                    native_status: status,
                });
            }
        }
    }
}

pub(crate) fn current_entered_device() -> Option<DeviceId> {
    THREAD_AUTHORITY.with(|authority| authority.borrow().entered_devices.last().copied())
}

pub(crate) fn reset_thread_authority_for_testing() {
    THREAD_AUTHORITY.with(|authority| {
        let mut authority = authority.borrow_mut();
        assert!(
            authority.entered_devices.is_empty(),
            "cannot reset CUDA thread authority while device entries remain live"
        );
        *authority = ThreadDeviceAuthority::default();
    });
}

pub(crate) type DeviceGuard = DeviceEntryGuard;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        runtime::DeviceHealthStatus,
        testing::device_safety::{DeviceEntryTestCall, DeviceSafetyTestScope, injected_device_id},
    };

    fn get(status: cudf_sys::cudf_status_t, device: i32) -> DeviceEntryTestCall {
        DeviceEntryTestCall::GetDevice { status, device }
    }

    fn set(target: i32, status: cudf_sys::cudf_status_t) -> DeviceEntryTestCall {
        DeviceEntryTestCall::SetDevice { expected_target: target, status }
    }

    fn detail<'a>(error: &'a Error, key: &str) -> Option<&'a str> {
        error.details().iter().find_map(|(candidate, value)| (*candidate == key).then_some(value.as_str()))
    }

    #[test]
    fn nested_cross_device_entry_is_rejected_before_switch_and_same_device_restore_is_top_level() {
        let scope = DeviceSafetyTestScope::install([
            get(cudf_sys::CUDF_SUCCESS, 1),
            set(0, cudf_sys::CUDF_SUCCESS),
            set(1, cudf_sys::CUDF_SUCCESS),
        ]);
        let owner = injected_device_id(0).expect("owner device");
        let ambient = injected_device_id(1).expect("ambient device");

        let outer = DeviceEntryGuard::enter(owner, "outer_entry").expect("outer entry");
        let error = DeviceEntryGuard::enter(ambient, "nested_cross_device")
            .expect_err("nested cross-device entry must be rejected before CUDA calls");
        assert_eq!(error.code(), ErrorCode::RuntimeBindingMismatch);
        assert_eq!(detail(&error, "reason"), Some("nested_cross_device_entry"));

        let nested = DeviceEntryGuard::enter(owner, "nested_same_device").expect("same-device nesting");
        drop(outer);
        assert_eq!(current_entered_device(), Some(owner));
        assert_eq!(scope.remaining_call_count(), 1, "outer drop must defer ambient restoration");
        drop(nested);

        assert_eq!(current_entered_device(), None);
        assert_eq!(scope.remaining_call_count(), 0);
        assert_eq!(scope.health_snapshot(owner).fault_count(), 0);
        assert_eq!(scope.health_snapshot(ambient).fault_count(), 0);
    }

    #[test]
    fn get_device_failure_is_reported_once_and_poisons_thread_authority() {
        let scope =
            DeviceSafetyTestScope::install([get(cudf_sys::CUDF_ERROR_FATAL_CUDA, 0), get(cudf_sys::CUDF_SUCCESS, 0)]);
        let owner = injected_device_id(0).expect("owner device");

        let error = DeviceEntryGuard::enter(owner, "get_device_failure").expect_err("get-device failure");
        assert_eq!(error.code(), ErrorCode::FatalCuda);
        let snapshot = scope.health_snapshot(owner);
        assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.fault_count(), 1);
        let first = snapshot.first_fault().expect("first get-device fault");
        assert_eq!(first.kind(), DeviceFaultKind::CurrentDeviceQueryFailed);
        assert_eq!(first.operation(), "get_device_failure");
        assert_eq!(first.native_status(), Some(cudf_sys::CUDF_ERROR_FATAL_CUDA as i32));

        let poisoned = DeviceEntryGuard::enter(owner, "after_get_device_failure")
            .expect_err("poisoned thread must reject entry before CUDA");
        assert_eq!(poisoned.code(), ErrorCode::FatalCuda);
        assert_eq!(detail(&poisoned, "reason"), Some("thread_device_authority_invalidated"));
        assert_eq!(scope.remaining_call_count(), 1);
        assert_eq!(scope.health_snapshot(owner).fault_count(), 1);
    }

    #[test]
    fn set_device_failure_is_reported_once_on_explicit_target_and_poisons_thread_authority() {
        let scope = DeviceSafetyTestScope::install([
            get(cudf_sys::CUDF_SUCCESS, 1),
            set(0, cudf_sys::CUDF_ERROR_FATAL_CUDA),
            get(cudf_sys::CUDF_SUCCESS, 1),
        ]);
        let owner = injected_device_id(0).expect("owner device");
        let ambient = injected_device_id(1).expect("ambient device");

        let error = DeviceEntryGuard::enter(owner, "set_device_failure").expect_err("set-device failure");
        assert_eq!(error.code(), ErrorCode::FatalCuda);
        let snapshot = scope.health_snapshot(owner);
        assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.fault_count(), 1);
        let first = snapshot.first_fault().expect("first set-device fault");
        assert_eq!(first.kind(), DeviceFaultKind::OwnerDeviceEntryFailed);
        assert_eq!(first.operation(), "set_device_failure");
        assert_eq!(first.native_status(), Some(cudf_sys::CUDF_ERROR_FATAL_CUDA as i32));
        assert_eq!(scope.health_snapshot(ambient).fault_count(), 0);

        let poisoned = DeviceEntryGuard::enter(ambient, "after_set_device_failure")
            .expect_err("poisoned thread must reject entry before CUDA");
        assert_eq!(detail(&poisoned, "fault_device"), Some("0"));
        assert_eq!(scope.remaining_call_count(), 1);
        assert_eq!(scope.health_snapshot(owner).fault_count(), 1);
    }

    #[test]
    fn restore_failure_is_reported_once_and_finally_invalidates_thread_authority() {
        let scope = DeviceSafetyTestScope::install([
            get(cudf_sys::CUDF_SUCCESS, 1),
            set(0, cudf_sys::CUDF_SUCCESS),
            set(1, cudf_sys::CUDF_ERROR_FATAL_CUDA),
            get(cudf_sys::CUDF_SUCCESS, 0),
        ]);
        let owner = injected_device_id(0).expect("owner device");
        let ambient = injected_device_id(1).expect("ambient device");

        let outer = DeviceEntryGuard::enter(owner, "outer_restore_entry").expect("outer entry");
        let nested = DeviceEntryGuard::enter(owner, "nested_restore_entry").expect("nested entry");
        drop(outer);
        assert_eq!(scope.remaining_call_count(), 2, "restoration must wait for the last same-device guard");
        drop(nested);

        let snapshot = scope.health_snapshot(ambient);
        assert_eq!(snapshot.status(), DeviceHealthStatus::Unhealthy);
        assert_eq!(snapshot.fault_count(), 1);
        let first = snapshot.first_fault().expect("first restore fault");
        assert_eq!(first.kind(), DeviceFaultKind::AmbientDeviceRestoreFailed);
        assert_eq!(first.operation(), "runtime::DeviceEntryGuard::drop");
        assert_eq!(first.native_status(), Some(cudf_sys::CUDF_ERROR_FATAL_CUDA as i32));
        assert_eq!(scope.health_snapshot(owner).fault_count(), 0);

        let poisoned = DeviceEntryGuard::enter(owner, "after_restore_failure")
            .expect_err("restore failure must reject later work before CUDA/options creation");
        assert_eq!(poisoned.code(), ErrorCode::FatalCuda);
        assert_eq!(detail(&poisoned, "fault_device"), Some("1"));
        assert_eq!(scope.remaining_call_count(), 1);
        assert_eq!(scope.health_snapshot(ambient).fault_count(), 1);
    }
}
