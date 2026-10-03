//! Typed failure evidence carried by a [`super::QueryError`].
//!
//! Code that decides (retry, repartition, cleanup, attempt reports) reads these
//! fields. [`super::QueryError::wire_facts`] renders them under the diagnostic
//! fact keys that reports, Flight, and TSV already use.

use std::borrow::Cow;

use crate::plan::{JoinSide, NodeId};

/// Typed evidence about why and where a query failed.
///
/// Every field is filled by the owner that observed it. Decoration keeps the
/// first value: an outer layer never replaces the original failure location
/// or allocator snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FailureContext {
    /// Allocator snapshot of the budget scope the failing operation ran in.
    pub memory: Option<MemoryFailure>,
    /// Execution boundary that owned the failed unit of work.
    pub boundary: Option<ResourceBoundary>,
    /// Native allocator exception family that reported an allocation failure.
    pub allocation_source: Option<AllocationSource>,
    /// High-water mark of a completed resident-graph cache scope, present only
    /// after that scope finished its fence and health proof. Not a wire fact.
    pub graph_cache_scope_peak_bytes: Option<u64>,
    /// An owner-reported cleanup wait reached its deadline.
    pub cleanup_deadline_elapsed: bool,
}

/// The budget scope a failing operation ran in.
///
/// A snapshot shows the scope the error passed through; it does not prove a
/// query-limit rejection. The error code carries the allocation attribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFailure {
    /// Stage boundary label of the budget request.
    pub boundary: Cow<'static, str>,
    /// Operation that ran under the budget scope.
    pub operation: Cow<'static, str>,
    /// Allocator counters of the scope's member, when sampling succeeded.
    pub usage: Option<MemoryUsage>,
}

/// Allocator counters of one query-domain member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Allocation limit the allocator enforced for the member.
    pub limit_bytes: u64,
    pub peak_bytes: u64,
    pub peak_aligned_bytes: u64,
    pub current_bytes: u64,
    pub current_aligned_bytes: u64,
    pub failed_allocation_count: u64,
    pub largest_failed_allocation_bytes: u64,
}

/// Execution boundary that owned a failed unit of work.
///
/// Labels are diagnostics. Retry and repartition decisions read `side`
/// together with the error code, never the labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBoundary {
    pub memory_boundary: &'static str,
    pub stage: &'static str,
    pub node_id: Option<NodeId>,
    /// Semantic join input whose working set failed.
    pub side: Option<JoinSide>,
    /// Where execution can resume after a local retry.
    pub resume: Option<&'static str>,
}

impl ResourceBoundary {
    /// A boundary whose memory boundary and stage share one label.
    pub const fn new(memory_boundary: &'static str, node_id: NodeId) -> Self {
        Self { memory_boundary, stage: memory_boundary, node_id: Some(node_id), side: None, resume: None }
    }

    #[must_use]
    pub const fn with_stage(mut self, stage: &'static str) -> Self {
        self.stage = stage;
        self
    }

    #[must_use]
    pub const fn with_side(mut self, side: JoinSide) -> Self {
        self.side = Some(side);
        self
    }

    #[must_use]
    pub const fn with_resume(mut self, resume: &'static str) -> Self {
        self.resume = Some(resume);
        self
    }
}

/// Native allocator exception family behind an allocation failure.
///
/// `RmmOutOfMemory` is the callback-backed path through the query memory
/// resource. The `bad_alloc` variants identify direct throwing paths; when
/// they traverse a resident cache allocation scope that scope publishes an
/// exact allocator event independently. `Unknown` stays fail-closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationSource {
    Unknown,
    RmmOutOfMemory,
    RmmBadAlloc,
    StdBadAlloc,
}

impl AllocationSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::RmmOutOfMemory => "rmm_out_of_memory",
            Self::RmmBadAlloc => "rmm_bad_alloc",
            Self::StdBadAlloc => "std_bad_alloc",
        }
    }
}

/// The only allocator cap mechanism; reported under `budget_enforcement`.
const MEMORY_ENFORCEMENT: &str = "query_domain_member";
/// Reported under `budget_status` for every error that passed a budget scope.
const MEMORY_STATUS: &str = "exceeded_or_lower_layer_rejected";

impl FailureContext {
    /// Wire keys and values rendered from the typed fields.
    pub(super) fn projection(&self) -> Vec<(&'static str, Cow<'_, str>)> {
        let mut facts = Vec::new();
        if let Some(memory) = &self.memory {
            facts.push(("budget_boundary", Cow::Borrowed(memory.boundary.as_ref())));
            facts.push(("budget_operation", Cow::Borrowed(memory.operation.as_ref())));
            facts.push(("budget_enforcement", Cow::Borrowed(MEMORY_ENFORCEMENT)));
            facts.push(("budget_status", Cow::Borrowed(MEMORY_STATUS)));
            if let Some(usage) = memory.usage {
                facts.extend(
                    [
                        ("budget_bytes", usage.limit_bytes),
                        ("budget_peak_bytes", usage.peak_bytes),
                        ("budget_peak_aligned_bytes", usage.peak_aligned_bytes),
                        ("budget_current_bytes", usage.current_bytes),
                        ("budget_current_aligned_bytes", usage.current_aligned_bytes),
                        ("budget_failed_allocation_count", usage.failed_allocation_count),
                        ("budget_largest_failed_allocation_bytes", usage.largest_failed_allocation_bytes),
                    ]
                    .map(|(key, value)| (key, Cow::Owned(value.to_string()))),
                );
            }
        }
        if let Some(boundary) = &self.boundary {
            facts.push(("memory_boundary", Cow::Borrowed(boundary.memory_boundary)));
            facts.push(("stage", Cow::Borrowed(boundary.stage)));
            if let Some(node_id) = boundary.node_id {
                facts.push(("node_id", Cow::Owned(node_id.to_string())));
            }
            if let Some(side) = boundary.side {
                facts.push(("side", Cow::Owned(side.to_string())));
            }
            if let Some(resume) = boundary.resume {
                facts.push(("resume_boundary", Cow::Borrowed(resume)));
            }
        }
        if let Some(source) = self.allocation_source {
            facts.push(("cugraph_allocation_source", Cow::Borrowed(source.as_str())));
        }
        facts
    }

    pub(super) fn projects(&self, key: &str) -> bool {
        self.projection().iter().any(|(projected, _)| *projected == key)
    }
}
