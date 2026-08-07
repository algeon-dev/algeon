use std::{
    fmt,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cudf::{Runtime, runtime::BindingExtensionLease};

use crate::error::{Result, check_cuvs, from_cudf, from_interop, invalid_state, unsupported};

const CREATE_OPERATION: &str = "CuvsResources::for_runtime";

struct NativeResources {
    raw: AtomicUsize,
    device: cudf::DeviceId,
    stream: usize,
    memory_resource: usize,
}

impl NativeResources {
    fn create(device: cudf::DeviceId, stream: usize, memory_resource: usize) -> Result<Self> {
        cuvs_sys::validate_runtime_version()
            .map_err(|error| invalid_state(CREATE_OPERATION, error.to_string()).set_source(error))?;
        let _device = cudf::runtime::DeviceEntryGuard::enter(device, CREATE_OPERATION)
            .map_err(|error| from_cudf(CREATE_OPERATION, error))?;
        let mut raw = 0;
        unsafe { check_cuvs(cuvs_sys::cuvsResourcesCreate(&mut raw), CREATE_OPERATION)? };
        if raw == 0 {
            return Err(invalid_state(CREATE_OPERATION, "cuVS returned a null resources handle"));
        }
        let set_stream =
            unsafe { cuvs_sys::cuvsStreamSet(raw, (stream as *mut std::ffi::c_void).cast::<cuvs_sys::CUstream_st>()) };
        if let Err(error) = check_cuvs(set_stream, CREATE_OPERATION) {
            unsafe {
                let _ = cuvs_sys::cuvsResourcesDestroy(raw);
            }
            return Err(error);
        }
        Ok(Self { raw: AtomicUsize::new(raw), device, stream, memory_resource })
    }

    fn raw(&self, operation: &'static str) -> Result<cuvs_sys::cuvsResources_t> {
        let raw = self.raw.load(Ordering::Acquire);
        if raw == 0 {
            return Err(invalid_state(operation, "the runtime-bound cuVS resources have been closed"));
        }
        Ok(raw)
    }
}

impl Drop for NativeResources {
    fn drop(&mut self) {
        let raw = self.raw.swap(0, Ordering::AcqRel);
        if raw == 0 {
            return;
        }
        if let Ok(_device) = cudf::runtime::DeviceEntryGuard::enter(self.device, "CuvsResources::drop") {
            unsafe {
                let _ = cuvs_sys::cuvsResourcesDestroy(raw);
            }
        }
    }
}

/// One serialized cuVS resource generation for an exact cudf runtime binding.
///
/// cuVS 26.06 exposes stream selection but no memory-resource setter. This
/// bridge therefore accepts only runtimes whose memory resource is cudf's
/// process-lifetime current [`cudf::DeviceMemoryManager`] resource. Stream
/// reservations remain supported because they route allocations by stream
/// through that same resource. Arbitrary output-only/custom runtime memory
/// resources are rejected before a native handle is created.
pub struct CuvsResources {
    runtime: Arc<Runtime>,
    native: BindingExtensionLease<NativeResources>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CuvsResources {
    pub fn for_runtime(runtime: Arc<Runtime>) -> Result<Self> {
        let binding = runtime.execution_binding().map_err(|error| from_interop(CREATE_OPERATION, error))?;
        let stream = binding.cuda_stream().as_raw() as usize;
        let memory_resource = binding.memory_resource().as_raw() as usize;

        let manager = runtime.device_memory_manager().map_err(|error| from_cudf(CREATE_OPERATION, error))?;
        let manager_runtime = Runtime::builder()
            .with_device(runtime.device())
            .with_stream(runtime.stream().clone())
            .with_memory_resource(manager.resource().clone())
            .build()
            .map_err(|error| from_cudf(CREATE_OPERATION, error))?;
        let manager_binding =
            manager_runtime.execution_binding().map_err(|error| from_interop(CREATE_OPERATION, error))?;
        let current_memory_resource = manager_binding.memory_resource().as_raw() as usize;
        if memory_resource != current_memory_resource {
            return Err(unsupported(
                CREATE_OPERATION,
                "cuVS 26.06 cannot bind an arbitrary runtime memory resource; use the device memory manager resource and stream reservations",
            )
            .with_context("runtime_memory_resource", format!("0x{memory_resource:x}"))
            .with_context("current_memory_resource", format!("0x{current_memory_resource:x}")));
        }

        let native = runtime
            .get_or_try_init_binding_extension(CREATE_OPERATION, || {
                NativeResources::create(runtime.device(), stream, memory_resource)
            })
            .map_err(|error| match error {
                cudf::runtime::BindingExtensionInitError::Lifecycle(error) => from_cudf(CREATE_OPERATION, error),
                cudf::runtime::BindingExtensionInitError::Factory(error) => error,
            })?;
        let resources = Self { runtime, native, _not_send_sync: PhantomData };
        resources.validate_exact_binding(CREATE_OPERATION)?;
        Ok(resources)
    }

    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    pub fn device(&self) -> cudf::DeviceId {
        self.runtime.device()
    }

    pub fn binding_id(&self) -> cudf::runtime::ExecutionBindingId {
        self.runtime.binding_id()
    }

    pub(crate) fn validate_exact_binding(&self, operation: &'static str) -> Result<()> {
        let binding = self.runtime.execution_binding().map_err(|error| from_interop(operation, error))?;
        let guard = self.native.enter(operation).map_err(|error| from_cudf(operation, error))?;
        let native = guard.value();
        let actual_device = binding.device();
        let actual_stream = binding.cuda_stream().as_raw() as usize;
        let actual_memory_resource = binding.memory_resource().as_raw() as usize;
        if actual_device != native.device.ordinal() as i32
            || actual_stream != native.stream
            || actual_memory_resource != native.memory_resource
        {
            return Err(invalid_state(operation, "the cudf runtime no longer matches the cuVS resource binding")
                .with_context("expected_device", native.device.ordinal())
                .with_context("actual_device", actual_device)
                .with_context("expected_stream", format!("0x{:x}", native.stream))
                .with_context("actual_stream", format!("0x{actual_stream:x}"))
                .with_context("expected_memory_resource", format!("0x{:x}", native.memory_resource))
                .with_context("actual_memory_resource", format!("0x{actual_memory_resource:x}")));
        }
        native.raw(operation).map(|_| ())
    }

    pub(crate) fn retain(&self) -> Self {
        Self { runtime: Arc::clone(&self.runtime), native: self.native.clone(), _not_send_sync: PhantomData }
    }

    pub(crate) fn with_native<T>(
        &self,
        operation: &'static str,
        call: impl FnOnce(cuvs_sys::cuvsResources_t) -> Result<T>,
    ) -> Result<T> {
        let binding = self.runtime.execution_binding().map_err(|error| from_interop(operation, error))?;
        let _device = cudf::runtime::DeviceEntryGuard::enter(self.device(), operation)
            .map_err(|error| from_cudf(operation, error))?;
        let guard = self.native.enter(operation).map_err(|error| from_cudf(operation, error))?;
        let native = guard.value();
        if binding.device() != native.device.ordinal() as i32
            || binding.cuda_stream().as_raw() as usize != native.stream
            || binding.memory_resource().as_raw() as usize != native.memory_resource
        {
            return Err(invalid_state(operation, "cuVS call rejected an execution-binding mismatch"));
        }
        call(native.raw(operation)?)
    }
}

impl fmt::Debug for CuvsResources {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CuvsResources")
            .field("device", &self.device())
            .field("binding_id", &self.binding_id())
            .finish_non_exhaustive()
    }
}
