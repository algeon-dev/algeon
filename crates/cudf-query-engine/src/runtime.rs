use std::sync::Arc;

use crate::error::{QueryError, Result};

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    runtime: Arc<cudf::Runtime>,
    memory_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
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
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionContextBuilder {
    runtime: Option<Arc<cudf::Runtime>>,
    device_ordinal: Option<usize>,
    memory_policy: MemoryResourcePolicy,
    memory_control_policy: MemoryControlPolicy,
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

    pub fn build(self) -> Result<ExecutionContext> {
        self.memory_control_policy.validate()?;

        if let Some(runtime) = self.runtime {
            return Ok(ExecutionContext {
                runtime,
                memory_policy: MemoryResourcePolicy::ExternalRuntime,
                memory_control_policy: self.memory_control_policy,
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
pub enum ReservationKind {
    SourceRead,
    ChunkOperator,
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
