//! Hidden execution-context scaffolding for Phase 1 contract tests.

use std::sync::Arc;

use crate::{error::Result, execution_domain::ExecutionDomain, memory::DeviceMemoryResource, stream::CudaStream};

/// Hidden borrowed execution-context wrapper used by Phase 1 contract tests.
#[derive(Clone, Copy, Default)]
#[doc(hidden)]
pub struct __DebugExecutionContext<'a> {
    stream: Option<&'a CudaStream>,
    mr: Option<&'a DeviceMemoryResource>,
}

impl<'a> __DebugExecutionContext<'a> {
    /// Create an empty borrowed override.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the effective stream for one operation.
    pub fn with_stream(mut self, stream: &'a CudaStream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Override the effective memory resource for one operation.
    pub fn with_memory_resource(mut self, mr: &'a DeviceMemoryResource) -> Self {
        self.mr = Some(mr);
        self
    }

    pub(crate) fn resolve_output_domain(&self, input_domain: &Arc<ExecutionDomain>) -> Result<Arc<ExecutionDomain>> {
        ExecutionDomain::resolve_with_borrowed_override(
            input_domain,
            self.stream,
            self.mr,
            "execution_context::resolve_output_domain",
        )
    }
}
