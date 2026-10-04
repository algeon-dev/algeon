use super::DeviceId;

/// Diagnostic summary of where an owned cudf value is retained for execution.
///
/// This is an observation surface for logging, adapters, and tests. It is not
/// an external compatibility predicate; use operation APIs such as
/// `copy_into_runtime(...)` to establish placement instead of comparing these
/// fields yourself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPlacement {
    /// CUDA device associated with the retained value.
    pub device: DeviceId,
    /// Diagnostic CUDA stream handle value.
    pub stream_debug: usize,
    /// Diagnostic memory-resource handle value.
    pub memory_resource_debug: usize,
}
