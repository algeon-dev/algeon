//! Narrow NVTX ranges owned by native graph-algorithm execution.

#[cfg(feature = "cugraph")]
use std::ffi::{CStr, c_char, c_int};

#[cfg(feature = "cugraph")]
unsafe extern "C" {
    fn nvtxRangePushA(message: *const c_char) -> c_int;
    fn nvtxRangePop() -> c_int;
}

/// Thread-local NVTX range that closes when its native execution phase exits.
///
/// NVTX records profiling annotations only; it is not a runtime event store
/// and it does not participate in execution or admission decisions.
#[cfg(feature = "cugraph")]
pub(super) struct NvtxRange;

#[cfg(feature = "cugraph")]
impl NvtxRange {
    pub(super) fn new(label: &'static CStr) -> Self {
        // NVTX accepts a borrowed NUL-terminated label and returns a
        // diagnostic nesting level only.
        unsafe { nvtxRangePushA(label.as_ptr()) };
        Self
    }
}

#[cfg(feature = "cugraph")]
impl Drop for NvtxRange {
    fn drop(&mut self) {
        // Match the push even during error unwinding.
        unsafe { nvtxRangePop() };
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_build_range(operation: &str) -> Option<NvtxRange> {
    (operation == "datafusion_nexus_cugraph_similarity")
        .then(|| NvtxRange::new(c"datafusion_nexus.cugraph_similarity.graph_build"))
}
