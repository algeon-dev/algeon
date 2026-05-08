//! Runtime, memory, boundary, and reservation contracts for native execution.
//!
//! This module is intentionally close to the public API because these values
//! control more than local execution choices. Policy enum string values,
//! reservation fact keys, and retry fact keys flow into tests, reports, and
//! adapter diagnostics. Treat those strings as stable diagnostic contracts:
//! change them only with the same care as a report schema change.

use std::{collections::BTreeMap, sync::Arc};

use crate::error::{QueryError, Result};

/// Immutable execution configuration shared by native query execution.
///
/// The context owns or shares the [`cudf::Runtime`] used for all device work in
/// a query. The runtime determines the CUDA device, stream, and memory resource;
/// the query engine layers its own chunk-planning, reservation, retry, and
/// boundary policies on top of that runtime. Cloning the context is cheap and
/// keeps the same runtime/cache handles.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Shared cudf runtime used for GPU work.
    ///
    /// Runtime ownership is explicit: callers may inject an `Arc` to share a
    /// stream, allocator, and device across query plans, or allow the builder to
    /// create a fresh runtime from the configured device/memory-resource policy.
    runtime: Arc<cudf::Runtime>,
    /// Runtime allocator construction policy recorded for diagnostics.
    ///
    /// This does not replace the allocator owned by an injected runtime. When an
    /// external runtime is supplied, the runtime's existing memory resource is
    /// authoritative and this field records [`MemoryResourcePolicy::ExternalRuntime`].
    memory_policy: MemoryResourcePolicy,
    /// Engine-level memory control contract for bounded execution.
    ///
    /// Bounded fields drive chunk planning, reservation admission, and
    /// retry/fan-in decisions. Source chunk byte and row-group settings are
    /// planning thresholds where row-group metadata permits exact splitting.
    memory_control_policy: MemoryControlPolicy,
    /// Execution boundary contract used by source reads, build/probe reuse, and
    /// result materialization.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Optional durable GPU hash-join build cache.
    ///
    /// Required when the boundary policy asks for durable build
    /// materialization. Query-scoped build materialization does not use this
    /// cache.
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
}

impl ExecutionContext {
    /// Starts a builder for an [`ExecutionContext`].
    pub fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    /// Builds a context around an owned cudf runtime.
    ///
    /// The resulting context records [`MemoryResourcePolicy::ExternalRuntime`]
    /// because the runtime, including its allocator and stream, has already been
    /// constructed by the caller.
    pub fn from_runtime(runtime: cudf::Runtime) -> Self {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    /// Builds a context around a shared cudf runtime.
    ///
    /// Use this when multiple plans or adapter layers must share the same CUDA
    /// stream, device placement, allocator policy, or boundary observer state.
    pub fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Self {
        Self {
            runtime,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            hash_join_build_cache: None,
        }
    }

    /// Returns the cudf runtime used for all GPU operations.
    pub fn runtime(&self) -> &cudf::Runtime {
        &self.runtime
    }

    /// Returns the CUDA stream owned by the runtime.
    pub fn stream(&self) -> &cudf::runtime::CudaStream {
        self.runtime.stream()
    }

    /// Returns the device memory resource owned by the runtime.
    pub fn memory_resource(&self) -> &cudf::core::memory::DeviceMemoryResource {
        self.runtime.memory_resource()
    }

    /// Returns a shared handle to the runtime.
    ///
    /// The caller receives the same runtime object, not a new runtime with
    /// copied configuration.
    pub fn shared_runtime(&self) -> Arc<cudf::Runtime> {
        Arc::clone(&self.runtime)
    }

    /// Reports the effective device placement derived from the runtime.
    pub fn device_placement(&self) -> DevicePlacement {
        DevicePlacement::SingleDevice { device_ordinal: self.runtime.device().ordinal() }
    }

    /// Returns the memory-resource construction policy recorded for this context.
    pub fn memory_policy(&self) -> MemoryResourcePolicy {
        self.memory_policy
    }

    /// Returns the engine-level memory control policy.
    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    /// Returns the execution boundary policy.
    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    /// Returns the durable GPU hash-join build cache, when configured.
    ///
    /// A clone of the `Arc` is returned so callers can retain the cache without
    /// borrowing the context.
    pub fn hash_join_build_cache(&self) -> Option<Arc<crate::exec::GpuHashJoinBuildCache>> {
        self.hash_join_build_cache.as_ref().map(Arc::clone)
    }
}

/// Builder for [`ExecutionContext`].
///
/// The builder separates runtime construction policy from execution policy. If
/// a runtime is supplied, it is used as-is and `device_ordinal` /
/// `memory_policy` no longer construct CUDA state. If no runtime is supplied,
/// the builder creates a cudf runtime from the requested device ordinal and
/// memory-resource policy.
#[derive(Debug, Clone, Default)]
pub struct ExecutionContextBuilder {
    /// Prebuilt runtime to use instead of constructing one.
    runtime: Option<Arc<cudf::Runtime>>,
    /// CUDA device ordinal used only when this builder constructs the runtime.
    device_ordinal: Option<usize>,
    /// Memory resource used only when this builder constructs the runtime.
    memory_policy: MemoryResourcePolicy,
    /// Engine-level bounded/unbounded memory contract.
    memory_control_policy: MemoryControlPolicy,
    /// Source/build/probe/result boundary contract.
    boundary_policy: ExecutionBoundaryPolicy,
    /// Optional durable hash-join build cache.
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
}

impl ExecutionContextBuilder {
    /// Creates a builder with default runtime and execution policies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses an already-constructed cudf runtime.
    ///
    /// This marks the memory-resource policy as
    /// [`MemoryResourcePolicy::ExternalRuntime`] because allocator/device setup
    /// has happened outside the query engine.
    pub fn with_runtime(mut self, runtime: cudf::Runtime) -> Self {
        self.runtime = Some(Arc::new(runtime));
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Uses an already-shared cudf runtime.
    ///
    /// Prefer this over building separate runtimes when an adapter needs one
    /// CUDA stream, allocator, or device policy across multiple executions.
    pub fn with_shared_runtime(mut self, runtime: Arc<cudf::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self.memory_policy = MemoryResourcePolicy::ExternalRuntime;
        self
    }

    /// Selects the CUDA device ordinal for runtime construction.
    ///
    /// Ignored when a runtime is supplied with [`Self::with_runtime`] or
    /// [`Self::with_shared_runtime`].
    pub fn with_device_ordinal(mut self, device_ordinal: usize) -> Self {
        self.device_ordinal = Some(device_ordinal);
        self
    }

    /// Selects the memory resource used when constructing a runtime.
    ///
    /// Ignored for an injected runtime except for the recorded
    /// `ExternalRuntime` diagnostic state.
    pub fn with_memory_policy(mut self, memory_policy: MemoryResourcePolicy) -> Self {
        self.memory_policy = memory_policy;
        self
    }

    /// Sets the engine-level memory control contract.
    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    /// Sets the execution boundary policy.
    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    /// Provides the cache required by durable GPU hash-join materialization.
    pub fn with_hash_join_build_cache(mut self, cache: Arc<crate::exec::GpuHashJoinBuildCache>) -> Self {
        self.hash_join_build_cache = Some(cache);
        self
    }

    /// Validates policies, constructs or records the runtime, and returns a context.
    ///
    /// Validation failures carry report-facing facts such as
    /// `missing_hash_join_build_cache`; do not churn those fact keys casually.
    pub fn build(self) -> Result<ExecutionContext> {
        self.memory_control_policy.validate()?;
        self.boundary_policy.validate(self.hash_join_build_cache.is_some())?;

        if let Some(runtime) = self.runtime {
            return Ok(ExecutionContext {
                runtime,
                memory_policy: MemoryResourcePolicy::ExternalRuntime,
                memory_control_policy: self.memory_control_policy,
                boundary_policy: self.boundary_policy,
                hash_join_build_cache: self.hash_join_build_cache,
            });
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
        Ok(ExecutionContext {
            runtime: Arc::new(runtime),
            memory_policy,
            memory_control_policy: self.memory_control_policy,
            boundary_policy: self.boundary_policy,
            hash_join_build_cache: self.hash_join_build_cache,
        })
    }
}

/// Effective device placement for a native execution context.
///
/// The engine currently executes each context on one CUDA device. This type is a
/// public contract so future multi-device placement can extend the model without
/// overloading raw runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DevicePlacement {
    /// All work for the context runs on one CUDA device ordinal.
    SingleDevice { device_ordinal: usize },
}

/// Policy used when this crate constructs the cudf runtime memory resource.
///
/// This policy is about allocator construction, not bounded execution. Bounded
/// chunk planning and retry behavior are controlled by [`MemoryControlPolicy`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryResourcePolicy {
    /// Use cudf's default CUDA memory resource.
    #[default]
    DefaultCuda,
    /// Build a pool memory resource with an initial and optional maximum size.
    Pool {
        /// Initial pool size in bytes requested from cudf-rs.
        initial_pool_size: usize,
        /// Optional maximum pool size in bytes requested from cudf-rs.
        maximum_pool_size: Option<usize>,
    },
    /// Runtime was supplied by the caller and already owns its memory resource.
    ExternalRuntime,
}

/// Engine-level memory control contract.
///
/// `Unbounded` preserves the original whole-plan execution behavior. `Bounded`
/// enables source chunk planning, reservation admission, and bounded retry
/// paths. The bounded fields are contracts consumed by execution code: the
/// device budget is enforced by reservation admission, while source chunk byte
/// and row-group settings are planning targets applied when row-group metadata
/// permits. A single row group can exceed the byte target, and a source without
/// usable row-group metadata may still be planned as one all-row-groups chunk.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryControlPolicy {
    /// Execute without query-engine chunking or reservation admission.
    #[default]
    Unbounded,
    /// Execute through bounded planning and reservation checks.
    Bounded {
        /// Total resident device-memory budget, in bytes, enforced by the
        /// query-engine reservation tracker.
        device_budget_bytes: u64,
        /// Source chunk byte target used for initial chunk planning and byte
        /// estimates when row-group size metadata permits. Retry splitting is
        /// driven by row-group subdivision and `min_retry_chunk_row_groups`, not
        /// by shrinking toward this byte target. This is not a guaranteed hard
        /// maximum for every emitted chunk: an individual row group larger than
        /// this target can still be admitted as a one-row-group chunk, and
        /// metadata-unavailable files can be planned as one all-row-groups
        /// chunk.
        max_source_chunk_bytes: u64,
        /// Row-group count target for source chunk planning when row groups are
        /// known and splittable. Metadata-unavailable files can still be planned
        /// as one all-row-groups chunk, so this is not a guaranteed hard maximum
        /// for every emitted chunk.
        max_row_groups_per_chunk: usize,
        /// Maximum attempts for bounded retry paths before surfacing failure
        /// facts. For source chunk retries this is the recursion depth for one
        /// split branch; split retries can create multiple branches, so this is
        /// not a global cap on total attempts across the whole split tree. For
        /// final merge and final sort retries, this bounds fan-in reduction
        /// attempts or stages.
        max_retry_attempts: usize,
        /// Row-group count at or below which retry code stops splitting the
        /// current source chunk. This prevents further splitting once the current
        /// chunk has no more than this many row groups; it does not guarantee
        /// every child chunk produced by an earlier split has at least this many
        /// row groups.
        min_retry_chunk_row_groups: usize,
    },
}

impl MemoryControlPolicy {
    /// Stable diagnostic string for the memory-control policy.
    ///
    /// These values appear in metrics, facts, and report assertions.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::Bounded { .. } => "bounded",
        }
    }

    /// Returns whether bounded execution contracts are active.
    pub const fn is_bounded(self) -> bool {
        matches!(self, Self::Bounded { .. })
    }

    /// Validates the bounded policy before runtime construction.
    ///
    /// Bounded fields must be nonzero because downstream planning and retry
    /// logic require positive thresholds. The emitted `policy`, `field`, and
    /// `reason` facts are stable diagnostics.
    pub fn validate(self) -> Result<()> {
        let Self::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = self
        else {
            return Ok(());
        };

        validate_nonzero_policy_field("device_budget_bytes", device_budget_bytes == 0)?;
        validate_nonzero_policy_field("max_source_chunk_bytes", max_source_chunk_bytes == 0)?;
        validate_nonzero_policy_field("max_row_groups_per_chunk", max_row_groups_per_chunk == 0)?;
        validate_nonzero_policy_field("max_retry_attempts", max_retry_attempts == 0)?;
        validate_nonzero_policy_field("min_retry_chunk_row_groups", min_retry_chunk_row_groups == 0)?;
        Ok(())
    }
}

impl std::fmt::Display for MemoryControlPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Boundary contract for how native execution crosses source, build, probe, and
/// result ownership boundaries.
///
/// The policy is copied into metrics and bounded-join reservation facts. The
/// display format and component `as_str()` values are diagnostic contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionBoundaryPolicy {
    /// How source data enters device execution.
    source_read: SourceReadBoundary,
    /// How hash-join build artifacts are materialized and retained.
    build_materialization: BuildMaterializationBoundary,
    /// How probe execution consumes reusable build artifacts.
    probe_execution: ProbeExecutionBoundary,
    /// How final results leave device execution.
    result_materialization: ResultMaterializationBoundary,
}

impl Default for ExecutionBoundaryPolicy {
    fn default() -> Self {
        Self {
            source_read: SourceReadBoundary::DirectDeviceLocalParquet,
            build_materialization: BuildMaterializationBoundary::QueryScopedGpuHashJoin,
            probe_execution: ProbeExecutionBoundary::ReusableGpuHashJoin,
            result_materialization: ResultMaterializationBoundary::ExplicitSink,
        }
    }
}

impl ExecutionBoundaryPolicy {
    /// Creates an explicit boundary policy.
    pub const fn new(
        source_read: SourceReadBoundary,
        build_materialization: BuildMaterializationBoundary,
        probe_execution: ProbeExecutionBoundary,
        result_materialization: ResultMaterializationBoundary,
    ) -> Self {
        Self { source_read, build_materialization, probe_execution, result_materialization }
    }

    /// Returns the configured source-read boundary.
    pub const fn source_read(self) -> SourceReadBoundary {
        self.source_read
    }

    /// Returns the configured build-materialization boundary.
    pub const fn build_materialization(self) -> BuildMaterializationBoundary {
        self.build_materialization
    }

    /// Returns the configured probe-execution boundary.
    pub const fn probe_execution(self) -> ProbeExecutionBoundary {
        self.probe_execution
    }

    /// Returns the configured result-materialization boundary.
    pub const fn result_materialization(self) -> ResultMaterializationBoundary {
        self.result_materialization
    }

    /// Returns this policy with a different build-materialization boundary.
    pub const fn with_build_materialization(mut self, build_materialization: BuildMaterializationBoundary) -> Self {
        self.build_materialization = build_materialization;
        self
    }

    /// Validates cross-field boundary requirements.
    ///
    /// Durable hash-join builds require a configured build cache. The fact key
    /// `reason=missing_hash_join_build_cache` is report-facing.
    pub fn validate(self, has_hash_join_build_cache: bool) -> Result<()> {
        match self.build_materialization {
            BuildMaterializationBoundary::QueryScopedGpuHashJoin => {},
            BuildMaterializationBoundary::DurableGpuHashJoin if has_hash_join_build_cache => {},
            BuildMaterializationBoundary::DurableGpuHashJoin => {
                return Err(QueryError::new(
                    crate::error::QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.boundary_policy.validate",
                    "durable GPU hash-join build materialization requires a hash-join build cache",
                )
                .with_fact("build_materialization_boundary", self.build_materialization)
                .with_fact("reason", "missing_hash_join_build_cache"));
            },
        }
        match self.probe_execution {
            ProbeExecutionBoundary::ReusableGpuHashJoin => {},
        }
        Ok(())
    }
}

impl std::fmt::Display for ExecutionBoundaryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "source_read={},build_materialization={},probe_execution={},result_materialization={}",
            self.source_read, self.build_materialization, self.probe_execution, self.result_materialization
        )
    }
}

/// Source-read boundary used by native execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceReadBoundary {
    /// Read supported local Parquet sources directly into device data frames.
    DirectDeviceLocalParquet,
}

impl SourceReadBoundary {
    /// Stable diagnostic string for this boundary.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectDeviceLocalParquet => "direct_device_local_parquet",
        }
    }
}

impl std::fmt::Display for SourceReadBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Hash-join build materialization boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildMaterializationBoundary {
    /// Build artifacts are scoped to one query execution.
    QueryScopedGpuHashJoin,
    /// Build artifacts may be retained in the configured durable build cache.
    DurableGpuHashJoin,
}

impl BuildMaterializationBoundary {
    /// Stable diagnostic string for this boundary.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QueryScopedGpuHashJoin => "query_scoped_gpu_hash_join",
            Self::DurableGpuHashJoin => "durable_gpu_hash_join",
        }
    }
}

impl std::fmt::Display for BuildMaterializationBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Probe execution boundary for native hash joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeExecutionBoundary {
    /// Probe execution can reuse an already-materialized GPU hash-join build.
    ReusableGpuHashJoin,
}

impl ProbeExecutionBoundary {
    /// Stable diagnostic string for this boundary.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReusableGpuHashJoin => "reusable_gpu_hash_join",
        }
    }
}

impl std::fmt::Display for ProbeExecutionBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result materialization boundary for native execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultMaterializationBoundary {
    /// Results are explicitly materialized through the configured sink boundary.
    ExplicitSink,
}

impl ResultMaterializationBoundary {
    /// Stable diagnostic string for this boundary.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitSink => "explicit_sink",
        }
    }
}

impl std::fmt::Display for ResultMaterializationBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Kind of memory reservation being admitted by bounded execution.
///
/// `as_str()` values are emitted in reservation events and failure facts. They
/// classify the memory boundary, not necessarily the concrete cudf allocation
/// that follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservationKind {
    /// Device memory needed while reading a source chunk.
    SourceRead,
    /// Temporary memory for a chunk-local operator.
    ChunkOperator,
    /// Resident output frame produced by an operator.
    OperatorOutput,
    /// Build-side input retained for a join.
    BuildSide,
    /// Probe-side input retained for a join.
    ProbeSide,
    /// Hash table storage for a reusable hash join.
    HashTable,
    /// Retained probe state for filtered semi/anti joins.
    FilteredProbeState,
    /// Retained size-data objects for mixed/residual/conditional joins.
    MixedSizeData,
    /// Join gather-map or index-vector storage.
    GatherMap,
    /// Materialized join output frame.
    JoinOutput,
    /// Intermediate aggregate or join pipeline partial.
    PartialState,
    /// Final aggregate merge workspace or output.
    FinalMerge,
    /// Final sort workspace or output.
    FinalSort,
    /// Final device-to-host materialization boundary.
    Materialization,
}

impl ReservationKind {
    /// Stable diagnostic string for this reservation kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SourceRead => "source_read",
            Self::ChunkOperator => "chunk_operator",
            Self::OperatorOutput => "operator_output",
            Self::BuildSide => "build_side",
            Self::ProbeSide => "probe_side",
            Self::HashTable => "hash_table",
            Self::FilteredProbeState => "filtered_probe_state",
            Self::MixedSizeData => "mixed_size_data",
            Self::GatherMap => "gather_map",
            Self::JoinOutput => "join_output",
            Self::PartialState => "partial_state",
            Self::FinalMerge => "final_merge",
            Self::FinalSort => "final_sort",
            Self::Materialization => "materialization",
        }
    }
}

impl std::fmt::Display for ReservationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Status of a bounded join reservation requirement.
///
/// Status strings flow into bounded-join reservation facts. `Modeled` means the
/// engine has an explicit reservation path for that boundary; other statuses are
/// kept as reportable blockers/deferred work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundedJoinReservationStatus {
    /// The reservation is modeled and enforced by current bounded execution.
    Modeled,
    /// The reservation depends on a lower-level cudf-rs/libcudf contract that is
    /// not available yet.
    BlockedByLowerLevel,
    /// The reservation is intentionally deferred until another prerequisite is
    /// available.
    DeferredUntilPrerequisite,
}

impl BoundedJoinReservationStatus {
    /// Stable diagnostic string for this status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Modeled => "modeled",
            Self::BlockedByLowerLevel => "blocked_by_lower_level",
            Self::DeferredUntilPrerequisite => "deferred_until_prerequisite",
        }
    }
}

impl std::fmt::Display for BoundedJoinReservationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One row in the bounded join reservation contract.
///
/// Each requirement identifies a query-engine memory boundary plus the
/// observability or admission contract needed to bound it safely. Some modeled
/// requests are planning/admission contracts backed by metadata estimates or
/// configured thresholds rather than exact device-memory sizing. The
/// `memory_boundary`, `required_contract`, and `detail` strings are
/// report-facing diagnostics; avoid renaming them unless downstream reports and
/// tests are updated deliberately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedJoinReservationRequirement {
    /// Reservation category enforced by bounded execution.
    reservation_kind: ReservationKind,
    /// Stable memory-boundary name used in facts and reservation requests.
    memory_boundary: &'static str,
    /// Query-engine or cudf-rs contract required to observe or admit the boundary.
    required_contract: &'static str,
    /// Current modeling status for this boundary.
    status: BoundedJoinReservationStatus,
    /// Stable detail string explaining how the requirement is modeled or blocked.
    detail: &'static str,
}

impl BoundedJoinReservationRequirement {
    /// Creates a static bounded join reservation requirement.
    pub const fn new(
        reservation_kind: ReservationKind,
        memory_boundary: &'static str,
        required_contract: &'static str,
        status: BoundedJoinReservationStatus,
        detail: &'static str,
    ) -> Self {
        Self { reservation_kind, memory_boundary, required_contract, status, detail }
    }

    /// Reservation category for this requirement.
    pub const fn reservation_kind(self) -> ReservationKind {
        self.reservation_kind
    }

    /// Stable memory-boundary name for this requirement.
    pub const fn memory_boundary(self) -> &'static str {
        self.memory_boundary
    }

    /// Stable lower-level or query-engine contract string for this requirement.
    pub const fn required_contract(self) -> &'static str {
        self.required_contract
    }

    /// Current modeling status for this requirement.
    pub const fn status(self) -> BoundedJoinReservationStatus {
        self.status
    }

    /// Stable explanatory detail string.
    pub const fn detail(self) -> &'static str {
        self.detail
    }
}

/// Report-facing bounded join reservation requirement catalogue.
///
/// This list feeds contract tests, fallback reports, and unsupported-shape facts.
/// Ordering is part of the practical report contract because CSV helpers preserve
/// this order.
pub const BOUNDED_JOIN_RESERVATION_REQUIREMENTS: [BoundedJoinReservationRequirement; 11] = [
    BoundedJoinReservationRequirement::new(
        ReservationKind::SourceRead,
        "join_source_read",
        "query_engine_source_read_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_before_source_handoff",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::OperatorOutput,
        "join_operator_output",
        "query_engine_unary_output_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_bounded_resident_frame_ownership",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::BuildSide,
        "join_build_side",
        "query_engine_build_side_source_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_before_execution",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::HashTable,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_pre_build_sizing",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::ProbeSide,
        "join_probe_side",
        "query_engine_probe_side_source_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_ordered_bounded_join_execution_path",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::FilteredProbeState,
        "join_filtered_probe_state",
        "filtered_join_retained_probe_state",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_filtered_probe_state_observability",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::MixedSizeData,
        "join_mixed_size_data",
        "mixed_keyed_residual_or_conditional_full_retained_size_data",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_retained_join_size_data_observability",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::GatherMap,
        "join_gather_map",
        "join_gather_map_reservation_size_bytes",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_join_gather_map_sizing",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::JoinOutput,
        "join_output",
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::FinalSort,
        "join_pipeline_final_sort",
        "query_engine_final_sort_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_bounded_resident_frame_ownership",
    ),
    BoundedJoinReservationRequirement::new(
        ReservationKind::Materialization,
        "host_materialization",
        "query_engine_explicit_sink_materialization_reservation",
        BoundedJoinReservationStatus::Modeled,
        "modeled_by_explicit_sink_boundary",
    ),
];

/// Bounded join execution families emitted in reports and diagnostic facts.
///
/// `as_str()` values are contract strings consumed by tests, TSV reports, and
/// downstream diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundedJoinExecutionFamily {
    /// A single root equi join backed by a reusable hash table.
    RootReusableHashJoin,
    /// A single root semi/anti equi join backed by filtered probe state.
    RootFilteredSemiAntiJoin,
    /// A single root keyless residual-predicate join.
    KeylessConditionalJoin,
    /// A keyless conditional root join with bounded input subplans.
    KeylessConditionalJoinInputSubplanPipeline,
    /// A keyless conditional root join with aggregate-over-join subplan inputs.
    KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    /// A single root projected cross join.
    RootProjectedCrossJoin,
    /// A single root keyed join that also evaluates a residual predicate.
    RootMixedKeyedResidualJoin,
    /// A multi-join tree using reusable hash-join families.
    MultiJoinReusableHashPipeline,
    /// A multi-join tree that may include keyed residual inner/semi/anti joins.
    MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline,
}

impl BoundedJoinExecutionFamily {
    /// Stable diagnostic string for this execution family.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RootReusableHashJoin => "root_reusable_hash_join",
            Self::RootFilteredSemiAntiJoin => "root_filtered_semi_anti_join",
            Self::KeylessConditionalJoin => "keyless_conditional_join",
            Self::KeylessConditionalJoinInputSubplanPipeline => "keyless_conditional_join_input_subplan_pipeline",
            Self::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline => {
                "keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline"
            },
            Self::RootProjectedCrossJoin => "root_projected_cross_join",
            Self::RootMixedKeyedResidualJoin => "root_mixed_keyed_residual_join",
            Self::MultiJoinReusableHashPipeline => "multi_join_reusable_hash_pipeline",
            Self::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline => {
                "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline"
            },
        }
    }
}

/// Report-facing catalogue of bounded join execution families.
///
/// Ordering is preserved in CSV facts and tests.
pub const BOUNDED_JOIN_EXECUTION_FAMILIES: [BoundedJoinExecutionFamily; 9] = [
    BoundedJoinExecutionFamily::RootReusableHashJoin,
    BoundedJoinExecutionFamily::RootFilteredSemiAntiJoin,
    BoundedJoinExecutionFamily::KeylessConditionalJoin,
    BoundedJoinExecutionFamily::KeylessConditionalJoinInputSubplanPipeline,
    BoundedJoinExecutionFamily::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    BoundedJoinExecutionFamily::RootProjectedCrossJoin,
    BoundedJoinExecutionFamily::RootMixedKeyedResidualJoin,
    BoundedJoinExecutionFamily::MultiJoinReusableHashPipeline,
    BoundedJoinExecutionFamily::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline,
];

/// Public bounded join reservation contract and fact builder.
///
/// This type intentionally has no fields: it is a stable handle for the current
/// contract catalogue. Methods return report-facing strings and fact maps used
/// by contract tests and adapter diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BoundedJoinReservationContract;

impl BoundedJoinReservationContract {
    /// Returns the bounded join reservation requirements in report order.
    pub const fn requirements(self) -> &'static [BoundedJoinReservationRequirement] {
        &BOUNDED_JOIN_RESERVATION_REQUIREMENTS
    }

    /// Stable model-status fact value.
    pub const fn model_status(self) -> &'static str {
        "modeled"
    }

    /// Stable execution-status fact value for supported bounded join families.
    pub const fn execution_status(self) -> &'static str {
        "enabled"
    }

    /// Returns whether bounded join execution is enabled by this contract.
    pub const fn enabled(self) -> bool {
        true
    }

    /// Returns supported bounded join execution families in report order.
    pub const fn execution_families(self) -> &'static [BoundedJoinExecutionFamily] {
        &BOUNDED_JOIN_EXECUTION_FAMILIES
    }

    /// CSV of supported execution-family diagnostic strings.
    pub fn execution_families_csv(self) -> String {
        csv_join(self.execution_families().iter().map(|family| family.as_str()))
    }

    /// Human/report-facing unsupported-shape requirement text.
    pub fn supported_shape_requirement(self) -> String {
        format!("requires_one_of({})", self.execution_families_csv())
    }

    /// CSV of all reservation kinds in requirement order.
    pub fn reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| requirement.reservation_kind().as_str()))
    }

    /// CSV of `reservation_kind:status` pairs in requirement order.
    pub fn reservation_statuses_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| {
            format!("{}:{}", requirement.reservation_kind().as_str(), requirement.status().as_str())
        }))
    }

    /// CSV of reservation kinds that still block bounded execution.
    pub fn blocking_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() != BoundedJoinReservationStatus::Modeled)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    /// CSV of reservation kinds currently modeled by bounded execution.
    pub fn modeled_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::Modeled)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    /// CSV of reservation kinds deferred until another prerequisite lands.
    pub fn deferred_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::DeferredUntilPrerequisite)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    /// CSV of lower-level contracts that currently block bounded execution.
    pub fn lower_level_blockers_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::BlockedByLowerLevel)
                .then_some(requirement.required_contract())
        }))
    }

    /// CSV of `reservation_kind:required_contract:detail` rows.
    pub fn requirement_details_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| {
            format!(
                "{}:{}:{}",
                requirement.reservation_kind().as_str(),
                requirement.required_contract(),
                requirement.detail()
            )
        }))
    }

    /// Builds the bounded join reservation fact map.
    ///
    /// Fact keys and many values here are consumed by reports and tests. This
    /// map documents cudf-rs observability/reservation contracts that bounded
    /// execution relies on, including exact hash-table sizing, gather-map
    /// sizing, retained size-data observability, and explicit sink ownership.
    pub fn facts(self, boundary_policy: ExecutionBoundaryPolicy) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("boundary_policy", boundary_policy.to_string()),
            ("source_read_boundary", boundary_policy.source_read().to_string()),
            ("build_materialization_boundary", boundary_policy.build_materialization().to_string()),
            ("probe_execution_boundary", boundary_policy.probe_execution().to_string()),
            ("result_materialization_boundary", boundary_policy.result_materialization().to_string()),
            ("bounded_join_reservation_model", self.model_status().to_string()),
            ("bounded_join_execution_status", self.execution_status().to_string()),
            ("bounded_join_execution_families", self.execution_families_csv()),
            ("bounded_join_enabled", self.enabled().to_string()),
            ("bounded_join_reservation_kinds", self.reservation_kinds_csv()),
            ("bounded_join_reservation_statuses", self.reservation_statuses_csv()),
            ("blocked_reservations", self.blocking_reservation_kinds_csv()),
            ("modeled_reservations", self.modeled_reservation_kinds_csv()),
            ("deferred_reservations", self.deferred_reservation_kinds_csv()),
            ("bounded_join_lower_level_blockers", self.lower_level_blockers_csv()),
            ("bounded_join_reservation_requirements", self.requirement_details_csv()),
            ("cudf_rs_hash_table_observability", "hash_table_device_allocated_size_bytes".to_string()),
            ("cudf_rs_gather_map_observability", "join_indices_device_allocated_size_bytes".to_string()),
            (
                "cudf_rs_hash_join_gather_map_reservation_contract",
                "hash_join_gather_map_reservation_size_bytes".to_string(),
            ),
            ("cudf_rs_hash_join_output_size_observability", "hash_join_exact_output_size".to_string()),
            ("cudf_rs_hash_join_probe_output_size_hint", "hash_join_probe_with_output_size".to_string()),
            ("cudf_rs_filtered_join_hash_table_observability", "filtered_join_device_allocated_size_bytes".to_string()),
            ("cudf_rs_filtered_join_probe_state_contract", "filtered_join_retained_probe_state".to_string()),
            (
                "cudf_rs_filtered_join_probe_state_observability",
                "filtered_join_probe_device_allocated_size_bytes".to_string(),
            ),
            ("cudf_rs_filtered_join_output_size_observability", "filtered_join_probe_state_output_size".to_string()),
            (
                "cudf_rs_filtered_join_probe_materialization_contract",
                "filtered_join_probe_state_materialize_indices".to_string(),
            ),
            ("cudf_rs_conditional_join_output_size_observability", "conditional_join_exact_output_size".to_string()),
            ("cudf_rs_conditional_join_output_size_hint", "conditional_join_indices_with_output_size".to_string()),
            ("cudf_rs_mixed_join_output_size_observability", "mixed_inner_left_join_size_data_output_size".to_string()),
            (
                "cudf_rs_mixed_join_size_data_observability",
                "mixed_inner_left_join_size_data_device_allocated_size_bytes".to_string(),
            ),
            (
                "cudf_rs_mixed_join_gather_map_reservation_contract",
                "mixed_inner_left_join_size_data_gather_map_reservation_size_bytes".to_string(),
            ),
            ("cudf_rs_mixed_join_output_size_hint", "mixed_inner_left_join_size_data_materialize_indices".to_string()),
            (
                "cudf_rs_mixed_join_size_data_alias_compatibility",
                "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data".to_string(),
            ),
            ("cudf_rs_mixed_join_remaining_blockers", "".to_string()),
            (
                "bounded_join_mixed_keyed_residual_capability",
                "root_mixed_keyed_residual_join".to_string(),
            ),
            (
                "bounded_join_mixed_keyed_residual_capability_detail",
                "mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti".to_string(),
            ),
            (
                "bounded_join_mixed_keyed_residual_multi_join_capability",
                "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline".to_string(),
            ),
            (
                "bounded_join_mixed_keyed_residual_multi_join_capability_detail",
                "mixed_keyed_residual_multi_join_types=inner,left_semi,left_anti".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_input_subplan_pipeline_capability",
                "keyless_conditional_join_input_subplan_pipeline".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_input_subplan_pipeline_capability_detail",
                "root_join=keyless_conditional_inner;input_subplans=source_filter_projection_or_projection_wrapped_left_anti_hash_join_or_scalar_global_aggregate".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability",
                "keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline".to_string(),
            ),
            (
                "bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability_detail",
                "root_join=keyless_conditional_inner;left_input=projection_wrapped_grouped_aggregate_over_bounded_join_subplan;right_input=projection_wrapped_scalar_global_aggregate_over_bounded_join_subplan;root_output=left_only;post_root=projection_sort".to_string(),
            ),
            (
                "cudf_rs_mixed_join_pair_materialization_contract",
                "mixed_join_retained_size_data_materialize_indices".to_string(),
            ),
            (
                "cudf_rs_mixed_join_single_side_materialization_contract",
                "mixed_join_retained_size_data_materialize_single_indices".to_string(),
            ),
            (
                "cudf_rs_projected_cross_join_reservation_contract",
                "projected_cross_join_reservation_size_bytes".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_output_size_observability",
                "conditional_full_join_retained_size_data_output_size".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_size_data_contract",
                "conditional_full_join_retained_size_data".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_gather_map_reservation_contract",
                "conditional_full_join_size_data_gather_map_reservation_size_bytes".to_string(),
            ),
            (
                "cudf_rs_conditional_full_join_materialization_contract",
                "conditional_full_join_retained_size_data_materialize_indices".to_string(),
            ),
            (
                "cudf_rs_dataframe_gather_output_reservation_contract",
                "dataframe_gather_output_reservation_size_bytes".to_string(),
            ),
            (
                "cudf_rs_hash_join_pre_build_reservation_contract",
                "hash_join_pre_build_reservation_size_bytes".to_string(),
            ),
            ("cudf_rs_hash_join_pre_build_reservation_status", "available_exact_flat_hash_join_builds".to_string()),
            ("missing_reservations", self.blocking_reservation_kinds_csv()),
        ])
    }

    /// Fact map attached to bounded join rejections.
    pub fn rejection_facts(self, boundary_policy: ExecutionBoundaryPolicy) -> BTreeMap<&'static str, String> {
        self.facts(boundary_policy)
    }

    /// Fact map for plans outside the supported bounded join shape catalogue.
    pub fn unsupported_shape_facts(self, boundary_policy: ExecutionBoundaryPolicy) -> BTreeMap<&'static str, String> {
        let mut facts = self.rejection_facts(boundary_policy);
        facts.insert("bounded_join_execution_status", "unsupported_plan_shape".to_string());
        facts.insert("bounded_join_unsupported_shape", self.supported_shape_requirement());
        facts
    }
}

fn csv_join<I, T>(values: I) -> String
where
    I: IntoIterator<Item = T>,
    T: ToString,
{
    let mut output = String::new();
    for value in values {
        if !output.is_empty() {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}

/// Reservation request submitted to bounded execution admission.
///
/// A request describes capacity submitted before retaining or committing a
/// bounded memory boundary. Some requests happen before invoking a cudf
/// operation; retained-output requests may happen after an operation, once
/// output bytes are observed. The internal reservation tracker grants a request
/// only if it fits in the remaining bounded budget. The returned internal guard
/// (`ActiveReservation`) represents admitted capacity and releases that capacity
/// either when explicitly released or when dropped.
///
/// Core reservation fields are recorded in metrics, and optional context is
/// included in failure facts when present. Keep field names, fact keys, and
/// memory-boundary strings stable unless report consumers are migrated
/// deliberately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReservationRequest {
    /// Category of memory being reserved.
    pub reservation_kind: ReservationKind,
    /// Stable boundary name for diagnostics.
    ///
    /// Defaults to `reservation_kind.as_str()` but callers should override it
    /// when a more precise boundary is part of the contract, such as
    /// `join_hash_table` or `host_materialization`.
    pub memory_boundary: String,
    /// Bytes requested for admission.
    pub requested_bytes: u64,
    /// Total bounded device budget used for this admission check.
    pub budget_bytes: u64,
    /// Optional plan node id associated with this reservation.
    pub node_id: Option<String>,
    /// Optional execution stage, such as a bounded join or retry stage.
    pub stage: Option<String>,
    /// Optional source chunk index.
    pub chunk_index: Option<usize>,
    /// Optional source file index.
    pub file_index: Option<usize>,
    /// Optional source path.
    pub path: Option<String>,
    /// Optional source row groups covered by the request.
    pub row_groups: Option<Vec<usize>>,
    /// Optional retry attempt number.
    pub attempt: Option<usize>,
    /// Optional maximum retry attempts for the current retry path.
    pub max_attempts: Option<usize>,
}

impl MemoryReservationRequest {
    /// Creates a reservation request for a kind, requested byte count, and budget.
    pub fn new(reservation_kind: ReservationKind, requested_bytes: u64, budget_bytes: u64) -> Self {
        Self {
            reservation_kind,
            memory_boundary: reservation_kind.as_str().to_string(),
            requested_bytes,
            budget_bytes,
            node_id: None,
            stage: None,
            chunk_index: None,
            file_index: None,
            path: None,
            row_groups: None,
            attempt: None,
            max_attempts: None,
        }
    }

    /// Overrides the stable memory-boundary diagnostic string.
    #[must_use]
    pub fn with_memory_boundary(mut self, memory_boundary: impl Into<String>) -> Self {
        self.memory_boundary = memory_boundary.into();
        self
    }

    /// Attaches a plan node id to the request.
    #[must_use]
    pub fn with_node_id(mut self, node_id: impl ToString) -> Self {
        self.node_id = Some(node_id.to_string());
        self
    }

    /// Attaches an execution stage to the request.
    #[must_use]
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    /// Attaches a source chunk index to the request.
    #[must_use]
    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    /// Attaches a source file index to the request.
    #[must_use]
    pub fn with_file_index(mut self, file_index: usize) -> Self {
        self.file_index = Some(file_index);
        self
    }

    /// Attaches a source path to the request.
    #[must_use]
    pub fn with_path(mut self, path: impl ToString) -> Self {
        self.path = Some(path.to_string());
        self
    }

    /// Attaches source row groups to the request.
    #[must_use]
    pub fn with_row_groups(mut self, row_groups: impl Into<Vec<usize>>) -> Self {
        self.row_groups = Some(row_groups.into());
        self
    }

    /// Attaches retry attempt metadata to the request.
    #[must_use]
    pub fn with_attempt(mut self, attempt: usize, max_attempts: usize) -> Self {
        self.attempt = Some(attempt);
        self.max_attempts = Some(max_attempts);
        self
    }

    /// Builds the stable reservation-denial error for this request.
    ///
    /// `granted_bytes` is the capacity that remained when admission failed, not
    /// a partial grant. The emitted facts are report-facing and should stay in
    /// sync with reservation metrics.
    pub fn denied_error(&self, granted_bytes: u64) -> QueryError {
        let mut error = QueryError::new(
            crate::error::QueryErrorCode::MemoryReservationFailed,
            "cudf_query_engine.memory.reserve",
            "memory reservation failed before native cudf execution",
        )
        .with_fact("memory_boundary", &self.memory_boundary)
        .with_fact("reservation_kind", self.reservation_kind)
        .with_fact("requested_bytes", self.requested_bytes)
        .with_fact("granted_bytes", granted_bytes)
        .with_fact("budget_bytes", self.budget_bytes);

        if let Some(node_id) = &self.node_id {
            error = error.with_fact("node_id", node_id);
        }
        if let Some(stage) = &self.stage {
            error = error.with_fact("stage", stage);
        }
        if let Some(chunk_index) = self.chunk_index {
            error = error.with_fact("chunk_index", chunk_index);
        }
        if let Some(file_index) = self.file_index {
            error = error.with_fact("file_index", file_index);
        }
        if let Some(path) = &self.path {
            error = error.with_fact("path", path);
        }
        if let Some(row_groups) = &self.row_groups {
            error = error.with_fact("row_groups", comma_separated_usize(row_groups));
        }
        if let Some(attempt) = self.attempt {
            error = error.with_fact("attempt", attempt);
        }
        if let Some(max_attempts) = self.max_attempts {
            error = error.with_fact("max_attempts", max_attempts);
        }

        error
    }
}

/// Retry/resume facts attached to bounded retry failures.
///
/// These fields document what bounded execution can safely retry after a
/// reservation denial or retryable `CudfResourceExhausted` failure: the retry
/// action, resume boundary, already-committed chunks/partials, and merge fan-in.
/// The fact keys emitted by
/// [`Self::attach_to_error`] are stable diagnostics used by tests and higher
/// level error reports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetryFacts {
    /// Stable retry action string, for example splitting a chunk or reducing
    /// merge fan-in.
    pub retry_action: Option<String>,
    /// Stable boundary where execution can resume after a retry.
    pub resume_boundary: Option<String>,
    /// Number of source chunks already committed before the failure.
    pub committed_chunks: Option<usize>,
    /// Number of partial results already committed before the failure.
    pub committed_partials: Option<usize>,
    /// Merge level being processed when the failure occurred.
    pub merge_level: Option<usize>,
    /// Merge fan-in attempted when the failure occurred.
    pub merge_fan_in: Option<usize>,
}

impl RetryFacts {
    /// Creates empty retry facts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the stable retry-action diagnostic value.
    #[must_use]
    pub fn with_retry_action(mut self, retry_action: impl Into<String>) -> Self {
        self.retry_action = Some(retry_action.into());
        self
    }

    /// Sets the stable resume-boundary diagnostic value.
    #[must_use]
    pub fn with_resume_boundary(mut self, resume_boundary: impl Into<String>) -> Self {
        self.resume_boundary = Some(resume_boundary.into());
        self
    }

    /// Records committed source chunks.
    #[must_use]
    pub fn with_committed_chunks(mut self, committed_chunks: usize) -> Self {
        self.committed_chunks = Some(committed_chunks);
        self
    }

    /// Records committed partial results.
    #[must_use]
    pub fn with_committed_partials(mut self, committed_partials: usize) -> Self {
        self.committed_partials = Some(committed_partials);
        self
    }

    /// Records the merge level involved in a retryable failure.
    #[must_use]
    pub fn with_merge_level(mut self, merge_level: usize) -> Self {
        self.merge_level = Some(merge_level);
        self
    }

    /// Records the merge fan-in involved in a retryable failure.
    #[must_use]
    pub fn with_merge_fan_in(mut self, merge_fan_in: usize) -> Self {
        self.merge_fan_in = Some(merge_fan_in);
        self
    }

    /// Attaches all present retry facts to an error.
    ///
    /// This preserves the original error code/status and only adds structured
    /// facts. Fact keys are stable diagnostic contracts.
    pub fn attach_to_error(&self, mut error: QueryError) -> QueryError {
        if let Some(retry_action) = &self.retry_action {
            error = error.with_fact("retry_action", retry_action);
        }
        if let Some(resume_boundary) = &self.resume_boundary {
            error = error.with_fact("resume_boundary", resume_boundary);
        }
        if let Some(committed_chunks) = self.committed_chunks {
            error = error.with_fact("committed_chunks", committed_chunks);
        }
        if let Some(committed_partials) = self.committed_partials {
            error = error.with_fact("committed_partials", committed_partials);
        }
        if let Some(merge_level) = self.merge_level {
            error = error.with_fact("merge_level", merge_level);
        }
        if let Some(merge_fan_in) = self.merge_fan_in {
            error = error.with_fact("merge_fan_in", merge_fan_in);
        }
        error
    }
}

fn comma_separated_usize(values: &[usize]) -> String {
    let mut output = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}

fn validate_nonzero_policy_field(field: &'static str, is_zero: bool) -> Result<()> {
    if is_zero {
        return Err(QueryError::new(
            crate::error::QueryErrorCode::InvalidPlan,
            "cudf_query_engine.memory_policy.validate",
            "bounded memory-control policy fields must be greater than zero",
        )
        .with_fact("policy", "bounded")
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(())
}
