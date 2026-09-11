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
    parquet_pass_read_limit: u64,
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
    /// **Outputs-only semantics** (matching upstream libcudf's `mr`
    /// parameter): the override receives the operation's output allocations.
    /// libcudf temporaries resolve through the device's long-lived memory
    /// domain and are accounted to the operation's stream; for budgeting,
    /// attach a stream reservation instead of overriding the resource.
    ///
    /// `mr` must have been created on the same CUDA device as the primary
    /// input runtime, or the operation fails with a runtime-binding mismatch.
    /// This borrow does not replace the runtime's owned default allocator for
    /// later calls.
    pub fn with_memory_resource(mut self, mr: &'a DeviceMemoryResource) -> Self {
        self.mr = Some(mr);
        self
    }

    /// Bound the read and decompression scratch of Parquet reads in this
    /// operation to `limit` bytes. Zero means no limit.
    ///
    /// The rows returned are unchanged. The limit is a hint: a single row group
    /// that does not fit within it is still loaded.
    pub fn with_parquet_pass_read_limit(mut self, limit: u64) -> Self {
        self.parquet_pass_read_limit = limit;
        self
    }

    pub(crate) fn resolve_output_domain(&self, input_domain: &Arc<ExecutionDomain>) -> Result<Arc<ExecutionDomain>> {
        let domain = ExecutionDomain::resolve_with_borrowed_override(
            input_domain,
            self.stream,
            self.mr,
            "execution_overrides::resolve_output_domain",
        )?;
        Ok(ExecutionDomain::with_parquet_pass_read_limit(&domain, self.parquet_pass_read_limit))
    }
}
