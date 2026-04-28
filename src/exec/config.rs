use std::sync::Arc;

use cudf::Runtime;

use crate::error::{Error, Result as AdapterResult};

/// Execution resources used by cudf-backed GPU islands.
///
/// The default config builds a cudf [`Runtime`] when a partition executes.
/// Supplying a runtime lets callers make placement, stream, allocator, and
/// boundary-observer policy explicit before the adapter starts GPU work.
#[derive(Debug, Clone, Default)]
pub struct CudfExecutionConfig {
    runtime: Option<Arc<Runtime>>,
}

impl CudfExecutionConfig {
    /// Create an execution config that builds a default cudf runtime at execution time.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use one caller-provided runtime for GPU island execution.
    pub fn with_runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self
    }

    /// Use one shared caller-provided runtime for GPU island execution.
    pub fn with_shared_runtime(mut self, runtime: Arc<Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub(crate) fn runtime_for_partition(&self, _partition: usize) -> AdapterResult<Arc<Runtime>> {
        if let Some(runtime) = &self.runtime {
            return Ok(Arc::clone(runtime));
        }

        Runtime::builder().build().map(Arc::new).map_err(|error| {
            Error::from_cudf("datafusion_cudf.runtime.build", "failed to initialize cudf runtime", error)
        })
    }
}
