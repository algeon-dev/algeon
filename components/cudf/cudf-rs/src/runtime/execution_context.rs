//! Borrowed per-operation stream and memory-resource overrides.

use std::sync::Arc;

use super::{execution_domain::ExecutionDomain, stream::CudaStream};
use crate::core::{error::Result, memory::DeviceMemoryResource};

/// Borrowed per-operation stream and/or memory-resource override.
///
/// This type is not an owned runtime and does not become retained provenance
/// by itself. It selects the stream and/or memory resource used for one
/// supported operation, then the output keeps provenance resolved from the
/// primary input plus any same-device overrides applied for that call.
///
/// Use [`crate::Runtime`] to own workflow defaults. Use
/// `ExecutionOverrides` only when one supported operation should borrow a
/// different same-device stream and/or memory resource without changing the
/// runtime's default handles.
#[derive(Clone, Copy, Default)]
pub struct ExecutionOverrides<'a> {
    stream: Option<&'a CudaStream>,
    mr: Option<&'a DeviceMemoryResource>,
}

impl<'a> ExecutionOverrides<'a> {
    /// Create an empty borrowed override.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the effective stream for one operation.
    ///
    /// `stream` must have been created on the same CUDA device as the primary
    /// input runtime, or the operation fails with a runtime-binding mismatch.
    pub fn with_stream(mut self, stream: &'a CudaStream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Override the effective memory resource for one operation.
    ///
    /// `mr` must have been created on the same CUDA device as the primary
    /// input runtime, or the operation fails with a runtime-binding mismatch.
    /// This borrow does not replace the runtime's owned default allocator for
    /// later calls.
    pub fn with_memory_resource(mut self, mr: &'a DeviceMemoryResource) -> Self {
        self.mr = Some(mr);
        self
    }

    pub(crate) fn resolve_output_domain(&self, input_domain: &Arc<ExecutionDomain>) -> Result<Arc<ExecutionDomain>> {
        ExecutionDomain::resolve_with_borrowed_override(
            input_domain,
            self.stream,
            self.mr,
            "execution_overrides::resolve_output_domain",
        )
    }
}
