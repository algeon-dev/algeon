use std::{collections::BTreeMap, sync::Arc};

use crate::error::{QueryError, Result};

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    runtime: Arc<cudf::Runtime>,
    memory_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
    boundary_policy: ExecutionBoundaryPolicy,
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
}

impl ExecutionContext {
    pub fn builder() -> ExecutionContextBuilder {
        ExecutionContextBuilder::default()
    }

    pub fn from_runtime(runtime: cudf::Runtime) -> Self {
        Self::from_shared_runtime(Arc::new(runtime))
    }

    pub fn from_shared_runtime(runtime: Arc<cudf::Runtime>) -> Self {
        Self {
            runtime,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: MemoryControlPolicy::Unbounded,
            boundary_policy: ExecutionBoundaryPolicy::default(),
            hash_join_build_cache: None,
        }
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

    pub fn memory_control_policy(&self) -> MemoryControlPolicy {
        self.memory_control_policy
    }

    pub fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.boundary_policy
    }

    pub fn hash_join_build_cache(&self) -> Option<Arc<crate::exec::GpuHashJoinBuildCache>> {
        self.hash_join_build_cache.as_ref().map(Arc::clone)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionContextBuilder {
    runtime: Option<Arc<cudf::Runtime>>,
    device_ordinal: Option<usize>,
    memory_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
    boundary_policy: ExecutionBoundaryPolicy,
    hash_join_build_cache: Option<Arc<crate::exec::GpuHashJoinBuildCache>>,
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

    pub fn with_memory_control_policy(mut self, memory_control_policy: MemoryControlPolicy) -> Self {
        self.memory_control_policy = memory_control_policy;
        self
    }

    pub fn with_boundary_policy(mut self, boundary_policy: ExecutionBoundaryPolicy) -> Self {
        self.boundary_policy = boundary_policy;
        self
    }

    pub fn with_hash_join_build_cache(mut self, cache: Arc<crate::exec::GpuHashJoinBuildCache>) -> Self {
        self.hash_join_build_cache = Some(cache);
        self
    }

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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryControlPolicy {
    #[default]
    Unbounded,
    Bounded {
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    },
}

impl MemoryControlPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::Bounded { .. } => "bounded",
        }
    }

    pub const fn is_bounded(self) -> bool {
        matches!(self, Self::Bounded { .. })
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionBoundaryPolicy {
    source_read: SourceReadBoundary,
    build_materialization: BuildMaterializationBoundary,
    probe_execution: ProbeExecutionBoundary,
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
    pub const fn new(
        source_read: SourceReadBoundary,
        build_materialization: BuildMaterializationBoundary,
        probe_execution: ProbeExecutionBoundary,
        result_materialization: ResultMaterializationBoundary,
    ) -> Self {
        Self { source_read, build_materialization, probe_execution, result_materialization }
    }

    pub const fn source_read(self) -> SourceReadBoundary {
        self.source_read
    }

    pub const fn build_materialization(self) -> BuildMaterializationBoundary {
        self.build_materialization
    }

    pub const fn probe_execution(self) -> ProbeExecutionBoundary {
        self.probe_execution
    }

    pub const fn result_materialization(self) -> ResultMaterializationBoundary {
        self.result_materialization
    }

    pub const fn with_build_materialization(mut self, build_materialization: BuildMaterializationBoundary) -> Self {
        self.build_materialization = build_materialization;
        self
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceReadBoundary {
    DirectDeviceLocalParquet,
}

impl SourceReadBoundary {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildMaterializationBoundary {
    QueryScopedGpuHashJoin,
    DurableGpuHashJoin,
}

impl BuildMaterializationBoundary {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeExecutionBoundary {
    ReusableGpuHashJoin,
}

impl ProbeExecutionBoundary {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultMaterializationBoundary {
    ExplicitSink,
}

impl ResultMaterializationBoundary {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservationKind {
    SourceRead,
    ChunkOperator,
    OperatorOutput,
    BuildSide,
    ProbeSide,
    HashTable,
    FilteredProbeState,
    MixedSizeData,
    GatherMap,
    JoinOutput,
    PartialState,
    FinalMerge,
    FinalSort,
    Materialization,
}

impl ReservationKind {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundedJoinReservationStatus {
    Modeled,
    BlockedByLowerLevel,
    DeferredUntilPrerequisite,
}

impl BoundedJoinReservationStatus {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedJoinReservationRequirement {
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    required_contract: &'static str,
    status: BoundedJoinReservationStatus,
    detail: &'static str,
}

impl BoundedJoinReservationRequirement {
    pub const fn new(
        reservation_kind: ReservationKind,
        memory_boundary: &'static str,
        required_contract: &'static str,
        status: BoundedJoinReservationStatus,
        detail: &'static str,
    ) -> Self {
        Self { reservation_kind, memory_boundary, required_contract, status, detail }
    }

    pub const fn reservation_kind(self) -> ReservationKind {
        self.reservation_kind
    }

    pub const fn memory_boundary(self) -> &'static str {
        self.memory_boundary
    }

    pub const fn required_contract(self) -> &'static str {
        self.required_contract
    }

    pub const fn status(self) -> BoundedJoinReservationStatus {
        self.status
    }

    pub const fn detail(self) -> &'static str {
        self.detail
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundedJoinExecutionFamily {
    RootReusableHashJoin,
    RootFilteredSemiAntiJoin,
    KeylessConditionalJoin,
    RootProjectedCrossJoin,
    RootMixedInnerLeftJoin,
    MultiJoinReusableHashPipeline,
    MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline,
}

impl BoundedJoinExecutionFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RootReusableHashJoin => "root_reusable_hash_join",
            Self::RootFilteredSemiAntiJoin => "root_filtered_semi_anti_join",
            Self::KeylessConditionalJoin => "keyless_conditional_join",
            Self::RootProjectedCrossJoin => "root_projected_cross_join",
            Self::RootMixedInnerLeftJoin => "root_mixed_inner_left_join",
            Self::MultiJoinReusableHashPipeline => "multi_join_reusable_hash_pipeline",
            Self::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline => {
                "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline"
            },
        }
    }
}

pub const BOUNDED_JOIN_EXECUTION_FAMILIES: [BoundedJoinExecutionFamily; 7] = [
    BoundedJoinExecutionFamily::RootReusableHashJoin,
    BoundedJoinExecutionFamily::RootFilteredSemiAntiJoin,
    BoundedJoinExecutionFamily::KeylessConditionalJoin,
    BoundedJoinExecutionFamily::RootProjectedCrossJoin,
    BoundedJoinExecutionFamily::RootMixedInnerLeftJoin,
    BoundedJoinExecutionFamily::MultiJoinReusableHashPipeline,
    BoundedJoinExecutionFamily::MultiJoinMixedKeyedResidualInnerLeftSemiLeftAntiPipeline,
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BoundedJoinReservationContract;

impl BoundedJoinReservationContract {
    pub const fn requirements(self) -> &'static [BoundedJoinReservationRequirement] {
        &BOUNDED_JOIN_RESERVATION_REQUIREMENTS
    }

    pub const fn model_status(self) -> &'static str {
        "modeled"
    }

    pub const fn execution_status(self) -> &'static str {
        "enabled"
    }

    pub const fn enabled(self) -> bool {
        true
    }

    pub const fn execution_families(self) -> &'static [BoundedJoinExecutionFamily] {
        &BOUNDED_JOIN_EXECUTION_FAMILIES
    }

    pub fn execution_families_csv(self) -> String {
        csv_join(self.execution_families().iter().map(|family| family.as_str()))
    }

    pub fn supported_shape_requirement(self) -> String {
        format!("requires_one_of({})", self.execution_families_csv())
    }

    pub fn reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| requirement.reservation_kind().as_str()))
    }

    pub fn reservation_statuses_csv(self) -> String {
        csv_join(self.requirements().iter().map(|requirement| {
            format!("{}:{}", requirement.reservation_kind().as_str(), requirement.status().as_str())
        }))
    }

    pub fn blocking_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() != BoundedJoinReservationStatus::Modeled)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    pub fn modeled_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::Modeled)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    pub fn deferred_reservation_kinds_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::DeferredUntilPrerequisite)
                .then_some(requirement.reservation_kind().as_str())
        }))
    }

    pub fn lower_level_blockers_csv(self) -> String {
        csv_join(self.requirements().iter().filter_map(|requirement| {
            (requirement.status() == BoundedJoinReservationStatus::BlockedByLowerLevel)
                .then_some(requirement.required_contract())
        }))
    }

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
            (
                "bounded_join_execution_family_alias_compatibility",
                "root_mixed_inner_left_join_is_legacy_alias_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_root_joins".to_string(),
            ),
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
                "root_mixed_inner_left_join_extended_to_full_left_semi_left_anti".to_string(),
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

    pub fn rejection_facts(self, boundary_policy: ExecutionBoundaryPolicy) -> BTreeMap<&'static str, String> {
        self.facts(boundary_policy)
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReservationRequest {
    pub reservation_kind: ReservationKind,
    pub memory_boundary: String,
    pub requested_bytes: u64,
    pub budget_bytes: u64,
    pub node_id: Option<String>,
    pub stage: Option<String>,
    pub chunk_index: Option<usize>,
    pub file_index: Option<usize>,
    pub path: Option<String>,
    pub row_groups: Option<Vec<usize>>,
    pub attempt: Option<usize>,
    pub max_attempts: Option<usize>,
}

impl MemoryReservationRequest {
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

    #[must_use]
    pub fn with_memory_boundary(mut self, memory_boundary: impl Into<String>) -> Self {
        self.memory_boundary = memory_boundary.into();
        self
    }

    #[must_use]
    pub fn with_node_id(mut self, node_id: impl ToString) -> Self {
        self.node_id = Some(node_id.to_string());
        self
    }

    #[must_use]
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    #[must_use]
    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    #[must_use]
    pub fn with_file_index(mut self, file_index: usize) -> Self {
        self.file_index = Some(file_index);
        self
    }

    #[must_use]
    pub fn with_path(mut self, path: impl ToString) -> Self {
        self.path = Some(path.to_string());
        self
    }

    #[must_use]
    pub fn with_row_groups(mut self, row_groups: impl Into<Vec<usize>>) -> Self {
        self.row_groups = Some(row_groups.into());
        self
    }

    #[must_use]
    pub fn with_attempt(mut self, attempt: usize, max_attempts: usize) -> Self {
        self.attempt = Some(attempt);
        self.max_attempts = Some(max_attempts);
        self
    }

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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetryFacts {
    pub retry_action: Option<String>,
    pub resume_boundary: Option<String>,
    pub committed_chunks: Option<usize>,
    pub committed_partials: Option<usize>,
    pub merge_level: Option<usize>,
    pub merge_fan_in: Option<usize>,
}

impl RetryFacts {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_retry_action(mut self, retry_action: impl Into<String>) -> Self {
        self.retry_action = Some(retry_action.into());
        self
    }

    #[must_use]
    pub fn with_resume_boundary(mut self, resume_boundary: impl Into<String>) -> Self {
        self.resume_boundary = Some(resume_boundary.into());
        self
    }

    #[must_use]
    pub fn with_committed_chunks(mut self, committed_chunks: usize) -> Self {
        self.committed_chunks = Some(committed_chunks);
        self
    }

    #[must_use]
    pub fn with_committed_partials(mut self, committed_partials: usize) -> Self {
        self.committed_partials = Some(committed_partials);
        self
    }

    #[must_use]
    pub fn with_merge_level(mut self, merge_level: usize) -> Self {
        self.merge_level = Some(merge_level);
        self
    }

    #[must_use]
    pub fn with_merge_fan_in(mut self, merge_fan_in: usize) -> Self {
        self.merge_fan_in = Some(merge_fan_in);
        self
    }

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
