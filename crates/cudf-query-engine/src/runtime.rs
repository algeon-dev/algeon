use std::sync::Arc;

use crate::error::{QueryError, Result};

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    runtime: Arc<cudf::Runtime>,
    memory_policy: MemoryResourcePolicy,
}

impl ExecutionContext {
    pub fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    pub fn from_runtime(runtime: cudf::Runtime) -> Self {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    pub fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Self {
        Self { runtime, memory_policy: MemoryResourcePolicy::ExternalRuntime }
    }

    pub fn runtime(&self) -> &cudf::Runtime {
        &self.runtime
    }

    pub fn stream(&self) -> &cudf::runtime::CudaStream {
        self.runtime.stream()
    }

    pub fn memory_resource(&self) -> &cudf::core::memory::DeviceMemoryResource {
        self.runtime.memory_resource()
    }

    pub fn shared_runtime(&self) -> Arc<cudf::Runtime> {
        Arc::clone(&self.runtime)
    }

    pub fn device_placement(&self) -> DevicePlacement {
        DevicePlacement::SingleDevice { device_ordinal: self.runtime.device().ordinal() }
    }

    pub fn memory_policy(&self) -> MemoryResourcePolicy {
        self.memory_policy
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionContextBuilder {
    runtime: Option<Arc<cudf::Runtime>>,
    device_ordinal: Option<usize>,
    memory_policy: MemoryResourcePolicy,
}

impl ExecutionContextBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    pub fn with_memory_policy(mut self, memory_policy: MemoryResourcePolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    pub fn build(self) -> Result<ExecutionContext> {
        if let Some(runtime) = self.runtime {
            return Ok(ExecutionContext { runtime, memory_policy: MemoryResourcePolicy::ExternalRuntime });
        }

        let mut builder = cudf::Runtime::builder();
        if let Some(device_ordinal) = self.device_ordinal {
            let device = cudf::DeviceId::new(device_ordinal).map_err(|error| {
                QueryError::from_cudf("cudf_query_engine.runtime.device", "failed to resolve CUDA device", error)
            })?;
            builder = builder.with_device(device);
        }

        let memory_policy = match self.memory_policy {
            MemoryResourcePolicy::ExternalRuntime => MemoryResourcePolicy::DefaultCuda,
            other => other,
        };

        match memory_policy {
            MemoryResourcePolicy::DefaultCuda | MemoryResourcePolicy::ExternalRuntime => {},
            MemoryResourcePolicy::Pool { initial_pool_size, maximum_pool_size } => {
                builder = builder.with_pool_memory_resource(initial_pool_size, maximum_pool_size);
            },
        }

        let runtime = builder.build().map_err(|error| {
            QueryError::from_cudf("cudf_query_engine.runtime.build", "failed to initialize cudf runtime", error)
        })?;
        Ok(ExecutionContext { runtime: Arc::new(runtime), memory_policy })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DevicePlacement {
    SingleDevice { device_ordinal: usize },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryResourcePolicy {
    #[default]
    DefaultCuda,
    Pool {
        initial_pool_size: usize,
        maximum_pool_size: Option<usize>,
    },
    ExternalRuntime,
}
