//! Public execution-domain owner for explicit stream and allocator control.

use std::sync::Arc;

use crate::{
    error::{Error, ErrorKind, Result},
    execution_domain::ExecutionDomain,
    memory::DeviceMemoryResource,
    stream::CudaStream,
};

/// Owns one explicit execution domain for a `cudf-rs` pipeline.
#[derive(Clone)]
pub struct Runtime {
    domain: Arc<ExecutionDomain>,
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

    /// Build a runtime with one explicit execution domain.
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
                "runtime must not use the ambient-default execution domain",
            )
            .with_operation("runtime::RuntimeBuilder::build"));
        }
        Ok(Runtime { domain })
    }
}
