//! Process-wide CUDA profiler capture controls.

use crate::{Result, core::error};

/// Start CUDA profiler collection for the current process.
///
/// Profilers such as Nsight Systems can use this call as an explicit capture
/// boundary. Call [`cuda_profiler_stop`] after the workload has quiesced.
pub fn cuda_profiler_start() -> Result<()> {
    // SAFETY: the cudf-sys shim takes no pointers and translates the CUDA
    // status into the shared cudf status/error-message contract.
    error::check_status_in(unsafe { cudf_sys::cudf_cuda_profiler_start() }, "runtime::cuda_profiler_start")
}

/// Stop CUDA profiler collection for the current process.
pub fn cuda_profiler_stop() -> Result<()> {
    // SAFETY: the cudf-sys shim takes no pointers and translates the CUDA
    // status into the shared cudf status/error-message contract.
    error::check_status_in(unsafe { cudf_sys::cudf_cuda_profiler_stop() }, "runtime::cuda_profiler_stop")
}
