//! Internal CUDA/cudf runtime helpers shared across modules.

use crate::error::{Error, ErrorKind, Result};

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
        let mut current = 0;
        // SAFETY: cudf writes the current device id into `current`, which is a
        // valid mutable out-pointer for the duration of the call.
        check_cudf_status(unsafe { cudf_sys::cudf_cuda_get_device(&mut current) }, operation)?;

        if current != target {
            // SAFETY: `target` was validated by higher-level handle creation;
            // cudf only observes the device ordinal during this call.
            check_cudf_status(unsafe { cudf_sys::cudf_cuda_set_device(target) }, operation)?;
        }

        Ok(Self { previous: current, restore: current != target })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.restore {
            // SAFETY: `self.previous` came from a successful
            // `cudf_cuda_get_device` call in `switch_to`.
            let _ = unsafe { cudf_sys::cudf_cuda_set_device(self.previous) };
        }
    }
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
}
