use std::sync::Arc;

use cudf_query_engine::{
    ExecutionBoundaryPolicy, ExecutionContext, GpuHashJoinBuildCache, GpuLocalParquetSourceCache, MemoryControlPolicy,
    MemoryResourcePolicy, SourceSchedulerPolicy,
};

use crate::error::Error;

/// Adapter-side native execution configuration.
///
/// This type holds the DataFusion-facing knobs for native cuDF execution. It does not eagerly create
/// cuDF runtime state; `execution_context` builds the native `ExecutionContext` at execution time so
/// runtime injection, allocator setup, memory-control validation, boundary policy, and durable build
/// cache wiring happen at the DataFusion/native boundary.
#[derive(Debug, Clone)]
pub struct NativeCudfExecutionConfig {
    runtime: Option<Arc<cudf::Runtime>>,
    device_ordinal: Option<usize>,
    memory_resource_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
    boundary_policy: ExecutionBoundaryPolicy,
    source_scheduler_policy: SourceSchedulerPolicy,
    hash_join_build_cache: Option<Arc<GpuHashJoinBuildCache>>,
    local_parquet_source_cache: Option<Arc<GpuLocalParquetSourceCache>>,
}

impl NativeCudfExecutionConfig {
    /// Use the default allocator policy with no native bounded-memory execution controls.
    pub fn unbounded() -> Self {
        Self {
            runtime: None,
            device_ordinal: None,
            memory_resource_policy: MemoryResourcePolicy::DefaultCuda,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
        }
    }

    /// Use default runtime allocator construction with bounded native memory-control settings.
    ///
    /// The bounded fields are expected to be nonzero where the native engine requires positive
    /// limits. Validation is performed when the native `ExecutionContext` is built, which keeps this
    /// builder lightweight and reports native validation failures as adapter execution errors.
    pub fn bounded(
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    ) -> Self {
        Self {
            runtime: None,
            device_ordinal: None,
            memory_resource_policy: MemoryResourcePolicy::DefaultCuda,
            memory_control_policy: MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            },
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
        }
    }

    /// Own `runtime` and share it with native execution through an `Arc`.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self
    }

    /// Use a caller-shared runtime, preserving its device, stream, allocator, and observer state.
    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Select the device for lazily created runtimes.
    ///
    /// This setting only matters when no runtime is supplied with `with_runtime` or
    /// `with_shared_runtime`; injected runtimes keep their own device selection.
    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    /// Set the allocator construction policy used when native execution builds a runtime.
    pub fn with_memory_resource_policy(mut self, memory_resource_policy: MemoryResourcePolicy) -> Self {
        self.memory_resource_policy = memory_resource_policy;
        self
    }

    /// Set bounded execution controls such as chunk sizing, reservations, and retry limits.
    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    /// Set source, build, probe, and result boundary choices for native execution.
    ///
    /// `ExecutionBoundaryPolicy` controls where data may remain device-resident across source reads,
    /// hash-join build/probe work, and result materialization. Durable hash-join build boundaries
    /// require `with_hash_join_build_cache` so the native context has a cache to write to and probe.
    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    /// Set bounded source scheduling controls.
    pub fn with_source_scheduler_policy(mut self, source_scheduler_policy: SourceSchedulerPolicy) -> Self {
        self.source_scheduler_policy = source_scheduler_policy;
        self
    }

    /// Attach the durable GPU hash-join build cache required by durable build-boundary policy.
    pub fn with_hash_join_build_cache(mut self, cache: Arc<GpuHashJoinBuildCache>) -> Self {
        self.hash_join_build_cache = Some(cache);
        self
    }

    /// Attach a durable GPU local-Parquet source cache for repeated source scans.
    ///
    /// The native engine currently accepts this only with unbounded memory
    /// control; bounded cache admission is deferred until cache-resident bytes
    /// are represented in the reservation ledger.
    pub fn with_local_parquet_source_cache(mut self, cache: Arc<GpuLocalParquetSourceCache>) -> Self {
        self.local_parquet_source_cache = Some(cache);
        self
    }

    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.source_scheduler_policy
    }

    // DataFusion-to-native config boundary. Native context construction and validation errors are
    // converted here into adapter errors before any query execution starts.
    pub(super) fn execution_context(&self) -> crate::error::Result<ExecutionContext> {
        let mut builder = ExecutionContext::builder()
            .with_memory_policy(self.memory_resource_policy)
            .with_memory_control_policy(self.memory_control_policy)
            .with_boundary_policy(self.boundary_policy)
            .with_source_scheduler_policy(self.source_scheduler_policy);
        if let Some(runtime) = &self.runtime {
            builder = builder.with_shared_runtime(Arc::clone(runtime));
        }
        if let Some(device_ordinal) = self.device_ordinal {
            builder = builder.with_device_ordinal(device_ordinal);
        }
        if let Some(cache) = &self.hash_join_build_cache {
            builder = builder.with_hash_join_build_cache(Arc::clone(cache));
        }
        if let Some(cache) = &self.local_parquet_source_cache {
            builder = builder.with_local_parquet_source_cache(Arc::clone(cache));
        }
        builder.build().map_err(|error| {
            Error::from_native(
                "datafusion_cudf.native_exec.context",
                "failed to build native cudf execution context",
                error,
            )
        })
    }
}
