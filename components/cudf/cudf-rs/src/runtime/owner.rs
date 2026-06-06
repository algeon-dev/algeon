//! Public owner for explicit stream / allocator defaults and retained provenance seeding.

use std::sync::Arc;

use super::{execution_domain::ExecutionDomain, stream::CudaStream};
use crate::core::{
    error::{Error, ErrorKind, Result},
    memory::DeviceMemoryResource,
};

/// Owns one explicit retained provenance bundle for a `cudf-rs` pipeline.
#[derive(Clone)]
pub struct Runtime {
    domain: Arc<ExecutionDomain>,
}

// SAFETY: Runtime owns an immutable retained-provenance handle bundle. The
// underlying stream and memory-resource handles are thread-safe wrappers.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").field("retained_provenance", &self.__debug_retained_provenance_id()).finish()
    }
}

/// Builder for [`Runtime`].
#[derive(Default)]
pub struct RuntimeBuilder {
    stream: Option<CudaStream>,
    mr: Option<DeviceMemoryResource>,
}

impl Runtime {
    /// Start building a new explicit runtime.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Synchronize the runtime's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream().synchronize()
    }

    /// Access the runtime's owned CUDA stream.
    pub fn stream(&self) -> &CudaStream {
        self.domain.explicit_stream().expect("runtime always owns an explicit stream")
    }

    /// Access the runtime's owned device memory resource.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        self.domain.explicit_memory_resource().expect("runtime always owns an explicit memory resource")
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }

    #[doc(hidden)]
    pub fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }

    #[doc(hidden)]
    pub fn __debug_retained_provenance_id(&self) -> u64 {
        self.__debug_execution_domain_id()
    }
}

impl RuntimeBuilder {
    /// Override the runtime's owned stream.
    pub fn with_stream(mut self, stream: CudaStream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Override the runtime's owned device memory resource.
    pub fn with_memory_resource(mut self, mr: DeviceMemoryResource) -> Self {
        self.mr = Some(mr);
        self
    }

    /// Build a runtime with one explicit retained provenance bundle.
    pub fn build(self) -> Result<Runtime> {
        let stream = match self.stream {
            Some(stream) => stream,
            None => CudaStream::new()?,
        };
        let mr = match self.mr {
            Some(mr) => mr,
            None => DeviceMemoryResource::cuda()?,
        };
        let domain = ExecutionDomain::from_handles(stream, mr)?;
        if domain.debug_id() == 0 {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "runtime must not use the ambient-default retained provenance",
            )
            .with_operation("runtime::RuntimeBuilder::build"));
        }
        Ok(Runtime { domain })
    }
}
