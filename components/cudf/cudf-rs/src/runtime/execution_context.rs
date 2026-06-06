//! Borrowed per-operation execution override.

use std::sync::Arc;

use super::{execution_domain::ExecutionDomain, stream::CudaStream};
use crate::core::{error::Result, memory::DeviceMemoryResource};

/// Borrowed per-operation stream and/or allocator override.
#[derive(Clone, Copy, Default)]
pub struct ExecutionContext<'a> {
    stream: Option<&'a CudaStream>,
    mr: Option<&'a DeviceMemoryResource>,
}

impl<'a> ExecutionContext<'a> {
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

#[doc(hidden)]
pub type __DebugExecutionContext<'a> = ExecutionContext<'a>;
