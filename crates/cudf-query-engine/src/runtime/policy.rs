//! Public runtime policy and diagnostic value types.

use crate::error::{QueryError, Result};

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

/// Memory space accounted by bounded reservation events.
///
/// Phase 1 models only the device memory owned by the current cudf runtime.
/// The display form is a stable metric value and must not inherit debug output
/// from runtime placement types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemorySpaceId {
    /// CUDA device memory for one runtime device ordinal.
    Device { device_ordinal: usize },
}

impl std::fmt::Display for MemorySpaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Device { device_ordinal } => write!(f, "GPU:{device_ordinal}"),
        }
    }
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
