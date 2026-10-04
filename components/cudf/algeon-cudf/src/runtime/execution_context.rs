//! Where one GPU operation runs.

use std::sync::Arc;

use super::{execution_domain::ExecutionDomain, stream::CudaStream};
use crate::core::{error::Result, memory::DeviceMemoryResource};

/// Where one GPU operation runs: the stream and memory resource its outputs
/// use.
///
/// [`Exec::RETAINED`] keeps the primary input's retained stream and memory
/// resource. [`crate::Runtime::exec`] selects a runtime's handles, and
/// [`Exec::new`], [`Exec::with_stream`], and [`Exec::with_memory_resource`]
/// borrow other same-device handles for one operation. An `Exec` is not
/// retained provenance by itself: the output's provenance resolves from the
/// primary input plus the handles this value selects. When the selected
/// stream or memory resource differs from the input's, columns the operation
/// leaves unchanged are copied so the output keeps one provenance.
#[derive(Clone, Copy)]
pub struct Exec<'a> {
    stream: Option<&'a CudaStream>,
    mr: Option<&'a DeviceMemoryResource>,
}

impl<'a> Exec<'a> {
    /// Outputs keep the primary input's retained stream and memory resource.
    pub const RETAINED: Exec<'static> = Exec { stream: None, mr: None };

    /// Run on `stream` and allocate outputs from `mr`.
    pub fn new(stream: &'a CudaStream, mr: &'a DeviceMemoryResource) -> Self {
        Self { stream: Some(stream), mr: Some(mr) }
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

    pub(crate) fn resolve_output_domain(&self, input_domain: &Arc<ExecutionDomain>) -> Result<Arc<ExecutionDomain>> {
        ExecutionDomain::resolve_with_borrowed_override(
            input_domain,
            self.stream,
            self.mr,
            "execution_overrides::resolve_output_domain",
        )
    }
}
