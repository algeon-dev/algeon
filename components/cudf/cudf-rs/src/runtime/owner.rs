//! Public owner for explicit stream / allocator defaults and retained provenance seeding.

use std::sync::Arc;

use super::{device::DeviceId, device_guard::DeviceGuard, execution_domain::ExecutionDomain, stream::CudaStream};
use crate::core::{
    error::{Error, ErrorKind, Result},
    memory::DeviceMemoryResource,
};

/// Owns one explicit retained provenance bundle for a `cudf-rs` pipeline.
///
/// Each `Runtime` is pinned to a specific CUDA device (see
/// [`RuntimeBuilder::with_device`]).  All resources created through this
/// runtime live on that device.
///
/// **Multi-GPU execution guarantee:** when an operation is invoked through
/// this runtime (via `_with_runtime` methods), the runtime's device is
/// re-entered at the FFI boundary before any GPU work is enqueued. This
/// ensures that compute runs on the correct device regardless of the calling
/// thread's ambient CUDA device. The ambient device is restored after each
/// FFI call.
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
        f.debug_struct("Runtime")
            .field("device", &self.device())
            .field("retained_provenance", &self.__debug_execution_domain_id())
            .finish()
    }
}

/// Builder for [`Runtime`].
#[derive(Default)]
pub struct RuntimeBuilder {
    stream: Option<CudaStream>,
    mr: Option<DeviceMemoryResource>,
    device: Option<DeviceId>,
}

impl Runtime {
    /// Start building a new explicit runtime.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// The CUDA device this runtime is pinned to.
    pub fn device(&self) -> DeviceId {
        self.domain.device().expect("runtime always owns an explicit domain with a device")
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
    /// Pin the runtime to a specific CUDA device.
    ///
    /// If not called, the runtime uses the calling thread's current device
    /// at build time (typically device 0).
    pub fn with_device(mut self, device: DeviceId) -> Self {
        self.device = Some(device);
        self
    }

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
    ///
    /// If [`with_device`](Self::with_device) was called, resources are created
    /// on that device.  Otherwise the device is inferred from supplied handles,
    /// or defaults to the calling thread's current device.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - An explicit device is given and a supplied handle was created on a
    ///   different device.
    /// - No explicit device is given and both handles were supplied but created
    ///   on different devices.
    pub fn build(self) -> Result<Runtime> {
        let device = self.resolve_device()?;

        // Temporarily switch to the target device so that any auto-created
        // stream or memory resource lands on the correct GPU.
        let _guard = DeviceGuard::new(device)?;

        let stream = match self.stream {
            Some(stream) => stream,
            None => CudaStream::new()?,
        };
        let mr = match self.mr {
            Some(mr) => mr,
            None => DeviceMemoryResource::cuda()?,
        };
        let domain = ExecutionDomain::from_handles_on_device(device, stream, mr)?;
        if domain.debug_id() == 0 {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "runtime must not use the ambient-default retained provenance",
            )
            .with_operation("runtime::RuntimeBuilder::build"));
        }
        Ok(Runtime { domain })
    }

    /// Resolve the target device, validating handle provenance.
    fn resolve_device(&self) -> Result<DeviceId> {
        let op = "runtime::RuntimeBuilder::build";

        if let Some(target) = self.device {
            // Explicit device — validate supplied handles agree.
            if let Some(ref stream) = self.stream {
                if stream.device() != target {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "cross-device stream: stream was created on a different device than the runtime target",
                    )
                    .with_operation(op)
                    .with_context("target_device", target.ordinal().to_string())
                    .with_context("stream_device", stream.device().ordinal().to_string()));
                }
            }
            if let Some(ref mr) = self.mr {
                if mr.device() != target {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "cross-device memory resource: MR was created on a different device than the runtime target",
                    )
                    .with_operation(op)
                    .with_context("target_device", target.ordinal().to_string())
                    .with_context("mr_device", mr.device().ordinal().to_string()));
                }
            }
            Ok(target)
        } else {
            // No explicit device — infer from handles.
            match (&self.stream, &self.mr) {
                (Some(stream), Some(mr)) => {
                    if stream.device() != mr.device() {
                        return Err(Error::new(
                            ErrorKind::InvalidOperation,
                            "cross-device handles: stream and memory resource were created on different devices",
                        )
                        .with_operation(op)
                        .with_context("stream_device", stream.device().ordinal().to_string())
                        .with_context("mr_device", mr.device().ordinal().to_string()));
                    }
                    Ok(stream.device())
                },
                (Some(stream), None) => Ok(stream.device()),
                (None, Some(mr)) => Ok(mr.device()),
                (None, None) => DeviceId::current(),
            }
        }
    }
}
