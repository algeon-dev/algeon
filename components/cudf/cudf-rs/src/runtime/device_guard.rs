//! RAII guard for CUDA device selection.
//!
//! `DeviceGuard` saves the calling thread's current device, switches to the
//! target device, and restores the original device on drop.

use super::device::DeviceId;
use crate::core::error::{self, Result};

/// RAII guard that sets the CUDA device on creation and restores the previous
/// device on drop.
///
/// This is `pub(crate)` — users interact with device selection through
/// [`RuntimeBuilder::with_device`](super::RuntimeBuilder::with_device).
pub(crate) struct DeviceGuard {
    previous: i32,
}

impl DeviceGuard {
    /// Switch to `target` and save the current device for restoration on drop.
    ///
    /// If the calling thread is already on `target`, the `cudaSetDevice` call
    /// is skipped (no-op optimization).
    pub(crate) fn new(target: DeviceId) -> Result<Self> {
        let mut current: i32 = 0;
        error::check_status_in(
            unsafe { cudf_sys::cudf_cuda_get_device(&mut current) },
            "DeviceGuard::new(get_device)",
        )?;

        if current != target.ordinal_raw() {
            error::check_status_in(
                unsafe { cudf_sys::cudf_cuda_set_device(target.ordinal_raw()) },
                "DeviceGuard::new(set_device)",
            )?;
        }

        Ok(Self { previous: current })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        // Best-effort restore. If this fails (extremely unlikely — would mean
        // the CUDA runtime is in a bad state), we log a warning rather than
        // panicking in a destructor.
        let mut current: i32 = 0;
        let status = unsafe { cudf_sys::cudf_cuda_get_device(&mut current) };
        if status != cudf_sys::CUDF_SUCCESS {
            eprintln!("[cudf::DeviceGuard] warning: failed to query current device during drop");
            return;
        }

        if current != self.previous {
            let status = unsafe { cudf_sys::cudf_cuda_set_device(self.previous) };
            if status != cudf_sys::CUDF_SUCCESS {
                eprintln!("[cudf::DeviceGuard] warning: failed to restore device {} during drop", self.previous);
            }
        }
    }
}
