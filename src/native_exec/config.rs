//! Native execution settings copied into rewritten DataFusion plan nodes.
//!
//! The config is cloneable and cheap to pass through planning. cuDF runtime
//! construction, allocator setup, memory-control validation, and cache wiring
//! happen later when execution builds the native query context.

#![deny(missing_docs)]

use std::sync::{Arc, OnceLock};

use cudf_query_engine::{
    AdaptiveMemoryHistory, ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext, GpuHashJoinBuildCache,
    GpuLocalParquetSourceCache, MemoryControlPolicy, MemoryResourcePolicy, NativePipelineExecutionMode, QueryHandle,
    ServiceStreamPool, SourceSchedulerPolicy, StddevSampExecutionMode, TenantQueryContext,
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
    service_stream_pool: Option<ServiceStreamPool>,
    service_stream_pool_size: Option<usize>,
    cancellation: ExecutionCancellation,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
    stddev_samp_execution_mode: StddevSampExecutionMode,
    adaptive_memory_history: Option<Arc<AdaptiveMemoryHistory>>,
    hash_join_build_cache: Option<Arc<GpuHashJoinBuildCache>>,
    local_parquet_source_cache: Option<Arc<GpuLocalParquetSourceCache>>,
    query_service_handle: Option<Arc<QueryHandle>>,
}

/// Native execution context plus any service leases that must outlive execution.
#[derive(Debug)]
pub(crate) struct NativeCudfExecutionContext {
    context: ExecutionContext,
    _tenant_context: Option<TenantQueryContext>,
}

impl NativeCudfExecutionContext {
    fn standalone(context: ExecutionContext) -> Self {
        Self { context, _tenant_context: None }
    }

    fn tenant_bound(tenant_context: TenantQueryContext) -> Self {
        let context = tenant_context.context().clone();
        Self { context, _tenant_context: Some(tenant_context) }
    }

    pub(crate) fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

impl NativeCudfExecutionConfig {
    /// Fallback bounded device budget, in bytes, used when auto-sizing cannot
    /// read the primary GPU's total memory.
    pub const AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES: u64 = 256 * 1024 * 1024;
    /// Device-memory headroom, in bytes, reserved for libcudf and driver
    /// overhead when deriving a bounded budget from `nvidia-smi`.
    pub const AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES: u64 = 500 * 1024 * 1024;
    /// Default bounded source chunk byte target used by
    /// [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES: u64 = 64 * 1024 * 1024;
    /// Default bounded source row-group target used by
    /// [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
    /// Default bounded retry-attempt cap used by [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS: usize = 3;
    /// Default bounded retry floor, in row groups, used by
    /// [`Self::bounded_auto_sized`].
    pub const DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

    /// Use the default allocator policy with no native bounded-memory execution controls.
    pub fn unbounded() -> Self {
        Self {
            runtime: None,
            device_ordinal: None,
            memory_resource_policy: MemoryResourcePolicy::DefaultCuda,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            service_stream_pool: None,
            service_stream_pool_size: None,
            cancellation: ExecutionCancellation::new(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            adaptive_memory_history: None,
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
            query_service_handle: None,
        }
    }

    /// Use default runtime allocator construction with an auto-sized bounded
    /// memory-control profile.
    ///
    /// The device budget is `nvidia-smi --query-gpu=memory.total --id=0`
    /// minus [`Self::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES`]. If `nvidia-smi`
    /// is unavailable, fails, or returns an unexpected payload, the budget
    /// falls back to [`Self::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES`]. Source
    /// chunking uses a 64 MiB byte target, one row group per chunk, three retry
    /// attempts, and one row group as the retry floor.
    pub fn bounded_auto_sized() -> Self {
        Self::bounded(
            Self::auto_sized_device_budget_bytes(),
            Self::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            Self::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            Self::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            Self::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
        )
    }

    /// Returns the auto-sized bounded device budget in bytes.
    ///
    /// The value is derived from the primary GPU's reported total memory minus
    /// 500 MiB of libcudf/driver headroom. A 256 MiB fallback keeps bounded
    /// report and service defaults constructible on hosts without `nvidia-smi`
    /// or without a visible GPU.
    pub fn auto_sized_device_budget_bytes() -> u64 {
        static AUTO_SIZED_DEVICE_BUDGET_BYTES: OnceLock<u64> = OnceLock::new();
        *AUTO_SIZED_DEVICE_BUDGET_BYTES.get_or_init(detect_auto_sized_device_budget_bytes)
    }

    /// Use default runtime allocator construction with bounded native memory-control settings.
    ///
    /// Byte fields are device-byte budgets. `max_source_chunk_bytes` and
    /// `max_row_groups_per_chunk` are planning targets for source chunking, not
    /// absolute promises that every emitted chunk will be smaller. Positive
    /// value validation is performed when the native `ExecutionContext` is
    /// built, which keeps this builder lightweight and reports native
    /// validation failures as adapter execution errors.
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
            service_stream_pool: None,
            service_stream_pool_size: None,
            cancellation: ExecutionCancellation::new(),
            native_pipeline_execution_mode: NativePipelineExecutionMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            adaptive_memory_history: None,
            hash_join_build_cache: None,
            local_parquet_source_cache: None,
            query_service_handle: None,
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

    /// Use a caller-owned service stream pool for bounded source-read workers.
    pub fn with_service_stream_pool(mut self, service_stream_pool: ServiceStreamPool) -> Self {
        self.service_stream_pool = Some(service_stream_pool);
        self
    }

    /// Set the size of the default process-global service stream pool.
    ///
    /// Ignored when a pool is supplied with [`Self::with_service_stream_pool`].
    pub fn with_service_stream_pool_size(mut self, pool_size: usize) -> Self {
        self.service_stream_pool_size = Some(pool_size);
        self
    }

    /// Attach a cooperative cancellation flag observed by native execution.
    pub fn with_cancellation(mut self, cancellation: ExecutionCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Select strict bounded or adaptive native bounded-pipeline execution.
    pub fn with_native_pipeline_execution_mode(mut self, mode: NativePipelineExecutionMode) -> Self {
        self.native_pipeline_execution_mode = mode;
        self
    }

    /// Select the execution contract for logical STDDEV_SAMP aggregates.
    pub fn with_stddev_samp_execution_mode(mut self, mode: StddevSampExecutionMode) -> Self {
        self.stddev_samp_execution_mode = mode;
        self
    }

    /// Attach an adaptive-only memory-history store for adaptive admission/execution.
    ///
    /// Strict bounded admission does not read this store. Supplying it while strict mode is selected
    /// leaves strict reports deterministic and only makes the store available if the mode is later
    /// changed to adaptive.
    pub fn with_adaptive_memory_history(mut self, memory_history: Arc<AdaptiveMemoryHistory>) -> Self {
        self.adaptive_memory_history = Some(memory_history);
        self
    }

    /// Attach the durable GPU hash-join build cache required by durable build-boundary policy.
    pub fn with_hash_join_build_cache(mut self, cache: Arc<GpuHashJoinBuildCache>) -> Self {
        self.hash_join_build_cache = Some(cache);
        self
    }

    /// Attach a durable GPU local-Parquet source cache for repeated source scans.
    ///
    /// Bounded native execution charges cache-resident source bytes through the
    /// query-engine reservation ledger before source/operator admission.
    pub fn with_local_parquet_source_cache(mut self, cache: Arc<GpuLocalParquetSourceCache>) -> Self {
        self.local_parquet_source_cache = Some(cache);
        self
    }

    /// Execute through an admitted multi-tenant query-service handle.
    ///
    /// The handle supplies the service-owned CUDA stream pool, source-fetch
    /// gates, deployment IO capability, and tenant device-budget slice used by
    /// native execution. The handle must remain unfinished until every plan
    /// node built from this config has completed execution; dropping the final
    /// handle releases the service admission slot.
    pub fn with_query_service_handle(mut self, handle: Arc<QueryHandle>) -> Self {
        self.memory_resource_policy = MemoryResourcePolicy::ExternalRuntime;
        self.memory_control_policy = handle.memory_control_policy();
        self.boundary_policy = handle.boundary_policy();
        self.source_scheduler_policy = handle.source_scheduler_policy();
        self.native_pipeline_execution_mode = handle.native_pipeline_execution_mode();
        self.stddev_samp_execution_mode = handle.stddev_samp_execution_mode();
        self.query_service_handle = Some(handle);
        self
    }

    /// Memory-control policy copied into the lazily built native context.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    /// Boundary policy controlling source/build/probe/result residency.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    /// Source scheduling policy used by bounded native source reads.
    pub fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.source_scheduler_policy
    }

    /// Strict bounded or adaptive pipeline execution mode.
    pub fn native_pipeline_execution_mode(&self) -> NativePipelineExecutionMode {
        self.native_pipeline_execution_mode
    }

    /// Execution contract for logical STDDEV_SAMP aggregates.
    pub fn stddev_samp_execution_mode(&self) -> StddevSampExecutionMode {
        self.stddev_samp_execution_mode
    }

    /// Adaptive-only memory history store, if configured.
    pub fn adaptive_memory_history(&self) -> Option<Arc<AdaptiveMemoryHistory>> {
        self.adaptive_memory_history.as_ref().map(Arc::clone)
    }

    // DataFusion-to-native config boundary. Native context construction and validation errors are
    // converted here into adapter errors before any query execution starts.
    pub(crate) fn execution_context(&self) -> crate::error::Result<NativeCudfExecutionContext> {
        if let Some(handle) = &self.query_service_handle {
            return handle.acquire_context().map(NativeCudfExecutionContext::tenant_bound).map_err(|error| {
                Error::from_native(
                    "datafusion_cudf.native_exec.context",
                    "failed to acquire query service native cudf execution context",
                    error,
                )
            });
        }

        let mut builder = ExecutionContext::builder()
            .with_memory_policy(self.memory_resource_policy)
            .with_memory_control_policy(self.memory_control_policy)
            .with_boundary_policy(self.boundary_policy)
            .with_source_scheduler_policy(self.source_scheduler_policy)
            .with_cancellation(self.cancellation.clone())
            .with_native_pipeline_execution_mode(self.native_pipeline_execution_mode)
            .with_stddev_samp_execution_mode(self.stddev_samp_execution_mode);
        if let Some(pool) = &self.service_stream_pool {
            builder = builder.with_service_stream_pool(pool.clone());
        }
        if let Some(pool_size) = self.service_stream_pool_size {
            builder = builder.with_service_stream_pool_size(pool_size);
        }
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
        if let Some(memory_history) = &self.adaptive_memory_history {
            builder = builder.with_adaptive_memory_history(Arc::clone(memory_history));
        }
        builder.build().map(NativeCudfExecutionContext::standalone).map_err(|error| {
            Error::from_native(
                "datafusion_cudf.native_exec.context",
                "failed to build native cudf execution context",
                error,
            )
        })
    }
}

fn detect_auto_sized_device_budget_bytes() -> u64 {
    let output = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"])
        .output()
    {
        Ok(output) if output.status.success() => output.stdout,
        _ => return NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
    };
    let total_mib: u64 = match std::str::from_utf8(&output).ok().and_then(|value| value.trim().lines().next()) {
        Some(line) => match line.trim().parse() {
            Ok(value) => value,
            Err(_) => return NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
        },
        None => return NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
    };
    let total_bytes = total_mib.saturating_mul(1024 * 1024);
    total_bytes
        .checked_sub(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_HEADROOM_BYTES)
        .filter(|value| *value > 0)
        .unwrap_or(NativeCudfExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES)
}
