//! Internal CUDA/cudf runtime helpers shared across modules.

use std::cell::RefCell;
#[cfg(test)]
use std::collections::VecDeque;

use crate::error::{Error, ErrorKind, Result};

#[derive(Debug, Clone)]
struct DeviceRestoreFailure {
    previous: i32,
    status: String,
}

thread_local! {
    static DEVICE_GUARD_POISON: RefCell<Option<DeviceRestoreFailure>> = const { RefCell::new(None) };
}

fn check_device_guard_poison(operation: &'static str) -> Result<()> {
    DEVICE_GUARD_POISON.with(|poisoned| match poisoned.borrow().as_ref() {
        Some(failure) => Err(Error::new(ErrorKind::InvalidState, operation)
            .with_message("failed to restore CUDA device in a previous DeviceGuard::drop")
            .with_context("restore_device", failure.previous.to_string())
            .with_context("restore_status", failure.status.clone())
            .set_persistent()),
        None => Ok(()),
    })
}

fn record_device_restore_failure(previous: i32, status: cudf_sys::cudf_status_t) {
    DEVICE_GUARD_POISON.with(|poisoned| {
        *poisoned.borrow_mut() = Some(DeviceRestoreFailure { previous, status: status.to_string() });
    });
}

#[cfg(test)]
fn clear_device_guard_poison() {
    DEVICE_GUARD_POISON.with(|poisoned| {
        poisoned.borrow_mut().take();
    });
}

fn cudf_cuda_get_device(current: &mut i32) -> cudf_sys::cudf_status_t {
    #[cfg(test)]
    if let Some(status) = test_cuda_get_device(current) {
        return status;
    }

    // SAFETY: cudf writes the current device id into `current`, which is a
    // valid mutable out-pointer for the duration of the call.
    unsafe { cudf_sys::cudf_cuda_get_device(current) }
}

fn cudf_cuda_set_device(target: i32) -> cudf_sys::cudf_status_t {
    #[cfg(test)]
    if let Some(status) = test_cuda_set_device(target) {
        return status;
    }

    // SAFETY: higher-level callers validate the device ordinal before
    // switching, and cudf only observes it for the duration of the call.
    unsafe { cudf_sys::cudf_cuda_set_device(target) }
}

/// Convert a raw cudf status code into the crate's structured error type.
///
/// This is the narrow translation boundary for internal cudf CUDA helpers.
pub(crate) fn check_cudf_status(status: cudf_sys::cudf_status_t, operation: &'static str) -> Result<()> {
    if status == cudf_sys::CUDF_SUCCESS {
        return Ok(());
    }

    let error = match status {
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => Error::new(ErrorKind::OutOfMemory, operation)
            .with_message(format!("cudf helper failed with status {}", status))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {}", status))
            .set_temporary(),
        cudf_sys::CUDF_ERROR_FATAL_CUDA => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {}", status))
            .set_persistent(),
        _ => Error::new(ErrorKind::Unexpected, operation)
            .with_message(format!("cudf helper failed with status {}", status)),
    };
    Err(error)
}

/// Restores the previously active CUDA device when dropped.
///
/// Used to make temporary device switches exception-safe around cudf calls.
#[derive(Debug)]
pub(crate) struct DeviceGuard {
    previous: i32,
    restore: bool,
}

impl DeviceGuard {
    /// Switch to `target` for the duration of the returned guard.
    ///
    /// When the guard is dropped, the prior device is restored if a switch
    /// actually occurred.
    pub(crate) fn switch_to(target: i32, operation: &'static str) -> Result<Self> {
        Self::switch_to_impl(target, operation, true)
    }

    /// Switch to `target` for destructor paths that must still attempt
    /// resource cleanup even after a prior restore failure poisoned the
    /// calling thread.
    pub(crate) fn switch_to_for_drop(target: i32, operation: &'static str) -> Result<Self> {
        Self::switch_to_impl(target, operation, false)
    }

    fn switch_to_impl(target: i32, operation: &'static str, report_poison: bool) -> Result<Self> {
        if report_poison {
            check_device_guard_poison(operation)?;
        }

        let mut current = 0;
        check_cudf_status(cudf_cuda_get_device(&mut current), operation)?;

        if current != target {
            check_cudf_status(cudf_cuda_set_device(target), operation)?;
        }

        if report_poison {
            check_device_guard_poison(operation)?;
        }

        Ok(Self { previous: current, restore: current != target })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.restore {
            let status = cudf_cuda_set_device(self.previous);
            if status != cudf_sys::CUDF_SUCCESS {
                record_device_restore_failure(self.previous, status);
            }
        }
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) enum TestCudaCall {
    GetDevice { status: cudf_sys::cudf_status_t, device: i32 },
    SetDevice { expected_target: i32, status: cudf_sys::cudf_status_t },
}

#[cfg(test)]
thread_local! {
    static TEST_CUDA_CALLS: RefCell<Option<VecDeque<TestCudaCall>>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct TestCudaHooks;

#[cfg(test)]
impl TestCudaHooks {
    pub(crate) fn install(calls: Vec<TestCudaCall>) -> Self {
        clear_device_guard_poison();
        TEST_CUDA_CALLS.with(|queued| {
            let mut queued = queued.borrow_mut();
            assert!(queued.is_none(), "test CUDA hooks already installed");
            *queued = Some(VecDeque::from(calls));
        });
        Self
    }
}

#[cfg(test)]
impl Drop for TestCudaHooks {
    fn drop(&mut self) {
        TEST_CUDA_CALLS.with(|queued| {
            let remaining = queued.borrow_mut().take().expect("test CUDA hooks should be installed");
            assert!(remaining.is_empty(), "unused test CUDA calls remained: {:?}", remaining);
        });
        clear_device_guard_poison();
    }
}

#[cfg(test)]
fn test_cuda_get_device(current: &mut i32) -> Option<cudf_sys::cudf_status_t> {
    TEST_CUDA_CALLS.with(|queued| {
        let mut queued = queued.borrow_mut();
        let calls = queued.as_mut()?;
        match calls.pop_front().expect("missing test CUDA call") {
            TestCudaCall::GetDevice { status, device } => {
                if status == cudf_sys::CUDF_SUCCESS {
                    *current = device;
                }
                Some(status)
            },
            other => panic!("expected GetDevice test CUDA call, got {:?}", other),
        }
    })
}

#[cfg(test)]
fn test_cuda_set_device(target: i32) -> Option<cudf_sys::cudf_status_t> {
    TEST_CUDA_CALLS.with(|queued| {
        let mut queued = queued.borrow_mut();
        let calls = queued.as_mut()?;
        match calls.pop_front().expect("missing test CUDA call") {
            TestCudaCall::SetDevice { expected_target, status } => {
                assert_eq!(target, expected_target, "unexpected CUDA device target");
                Some(status)
            },
            other => panic!("expected SetDevice test CUDA call, got {:?}", other),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorStatus;

    #[test]
    fn check_cudf_status_maps_out_of_memory_to_temporary_oom() {
        let err = check_cudf_status(cudf_sys::CUDF_ERROR_OUT_OF_MEMORY, "test_op").expect_err("expected error");
        assert_eq!(err.kind(), ErrorKind::OutOfMemory);
        assert!(err.is_temporary());
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn check_cudf_status_maps_fatal_cuda_to_persistent_unexpected() {
        let err = check_cudf_status(cudf_sys::CUDF_ERROR_FATAL_CUDA, "test_op").expect_err("expected error");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.status(), ErrorStatus::Persistent);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn device_guard_restore_failure_poisons_later_switches() {
        let _reset = TestCudaHooks::install(vec![
            TestCudaCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 0 },
            TestCudaCall::SetDevice { expected_target: 1, status: cudf_sys::CUDF_SUCCESS },
            TestCudaCall::SetDevice { expected_target: 0, status: cudf_sys::CUDF_ERROR_FATAL_CUDA },
        ]);

        {
            let _guard = DeviceGuard::switch_to(1, "outer_switch").expect("switch should succeed");
        }

        let err = DeviceGuard::switch_to(0, "later_switch").expect_err("poisoned device state should fail");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.status(), ErrorStatus::Persistent);
        assert_eq!(err.operation(), "later_switch");
        assert!(err.to_string().contains("failed to restore CUDA device"));
        assert!(err.to_string().contains("restore_device: 0"));
        assert!(err.to_string().contains("restore_status"));
    }

    #[test]
    fn device_guard_successful_restore_does_not_poison_later_switches() {
        let _reset = TestCudaHooks::install(vec![
            TestCudaCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 0 },
            TestCudaCall::SetDevice { expected_target: 1, status: cudf_sys::CUDF_SUCCESS },
            TestCudaCall::SetDevice { expected_target: 0, status: cudf_sys::CUDF_SUCCESS },
            TestCudaCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 0 },
        ]);

        {
            let _guard = DeviceGuard::switch_to(1, "outer_switch").expect("switch should succeed");
        }

        let guard = DeviceGuard::switch_to(0, "later_switch").expect("successful restore should not poison");
        drop(guard);
    }

    #[test]
    fn device_guard_drop_switch_ignores_existing_poison() {
        let _reset = TestCudaHooks::install(vec![
            TestCudaCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: 3 },
            TestCudaCall::SetDevice { expected_target: 1, status: cudf_sys::CUDF_SUCCESS },
            TestCudaCall::SetDevice { expected_target: 3, status: cudf_sys::CUDF_SUCCESS },
        ]);
        record_device_restore_failure(0, cudf_sys::CUDF_ERROR_FATAL_CUDA);

        let guard =
            DeviceGuard::switch_to_for_drop(1, "drop_switch").expect("drop path should still attempt device switch");
        assert!(guard.restore);
        drop(guard);
    }
}
