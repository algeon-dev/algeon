//! NVTX ranges owned by native query execution.

#[cfg(feature = "cugraph")]
use std::ffi::CStr;
use std::ffi::{CString, c_char, c_int};

use crate::{execution_contracts::DevicePlacement, observability::AttemptId, runtime::ExecutionContext};

unsafe extern "C" {
    fn nvtxRangePushA(message: *const c_char) -> c_int;
    fn nvtxRangePop() -> c_int;
}

/// Thread-local NVTX range that closes when its native execution phase exits.
///
/// NVTX records profiling annotations only; it is not a runtime event store
/// and it does not participate in execution or admission decisions.
pub(in crate::exec) struct NvtxRange {
    // Keep dynamic labels alive through the corresponding pop even though the
    // NVTX API consumes the pointer at range-push time.
    _label: CString,
}

impl NvtxRange {
    pub(in crate::exec) fn new(label: CString) -> Self {
        // NVTX accepts a borrowed NUL-terminated label and returns a
        // diagnostic nesting level only.
        unsafe { nvtxRangePushA(label.as_ptr()) };
        Self { _label: label }
    }

    #[cfg(feature = "cugraph")]
    pub(in crate::exec) fn static_label(label: &'static CStr) -> Self {
        Self::new(label.to_owned())
    }
}

impl Drop for NvtxRange {
    fn drop(&mut self) {
        // Match the push even during error unwinding.
        unsafe { nvtxRangePop() };
    }
}

/// Enters one attempt-owned native GPU execution range when the context was
/// admitted through the public attempt API.
pub(in crate::exec) fn attempt_execution_range(context: &ExecutionContext) -> Option<NvtxRange> {
    let attempt_id = context.attempt_id()?;
    let DevicePlacement::SingleDevice { device_ordinal } = context.device_placement() else {
        return None;
    };
    Some(NvtxRange::new(attempt_execution_label(attempt_id, device_ordinal)))
}

fn attempt_execution_label(attempt_id: AttemptId, device_ordinal: usize) -> CString {
    CString::new(format!(
        "algeon_datafusion.attempt_gpu_execution attempt_id={attempt_id} device_ordinal={device_ordinal}"
    ))
    .expect("engine-minted attempt identifiers and device ordinals never contain NUL")
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn graph_build_range(operation: &str) -> Option<NvtxRange> {
    graph_build_label(operation).map(NvtxRange::static_label)
}

#[cfg(feature = "cugraph")]
fn graph_build_label(operation: &str) -> Option<&'static CStr> {
    (operation == "algeon_datafusion_cugraph_similarity").then_some(c"algeon_datafusion.cugraph_similarity.graph_build")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attempt_execution_label_includes_the_engine_attempt_and_device() {
        let attempt_id = AttemptId::mint();
        let label = attempt_execution_label(attempt_id, 7);

        assert_eq!(
            label.to_str().expect("NVTX label is UTF-8"),
            format!("algeon_datafusion.attempt_gpu_execution attempt_id={attempt_id} device_ordinal=7")
        );
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn similarity_graph_build_label_remains_selective() {
        assert_eq!(
            graph_build_label("algeon_datafusion_cugraph_similarity"),
            Some(c"algeon_datafusion.cugraph_similarity.graph_build")
        );
        assert!(graph_build_label("algeon_datafusion_cugraph_pagerank").is_none());
    }
}
