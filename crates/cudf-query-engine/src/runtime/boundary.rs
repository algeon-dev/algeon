#![deny(missing_docs)]

use super::*;

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
