#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedSubplanMaterializationDiagnostics {
    /// Number of grouped aggregate nodes considered as possible materialization roots.
    pub considered_count: usize,
    /// Number of repeated aggregate families discovered.
    pub candidate_count: usize,
    /// Number of repeated aggregate families rewritten into shared materializations.
    pub applied_count: usize,
    /// Per-candidate evidence in optimizer order.
    pub candidates: Vec<SharedSubplanMaterializationCandidateDiagnostics>,
}

impl SharedSubplanMaterializationDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.candidate_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.candidate_count += other.candidate_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate repeated-subplan materialization evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedSubplanMaterializationCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub(crate) rule_name: &'static str,
    /// Stable fingerprint for the repeated aggregate family.
    pub(crate) family_key: String,
    /// Aggregate nodes that belong to this repeated family.
    pub aggregate_node_ids: Vec<NodeId>,
    /// Number of aggregate nodes in the repeated family.
    pub reuse_count: usize,
    /// Grouping expressions used by the aggregate family.
    pub group_exprs: Vec<String>,
    /// Aggregate expressions used by the aggregate family.
    pub aggregate_exprs: Vec<String>,
    /// Number of filtered aggregate expressions the shared materialization would create.
    pub filtered_aggregate_count: usize,
    /// Memory-shape policy used for shared filtered aggregate materialization.
    pub filtered_aggregate_memory_shape: String,
    /// Row-count estimate used for shared filtered aggregate materialization, if known.
    pub filtered_aggregate_estimation_input_rows: Option<u64>,
    /// Estimated single-pass filtered-input materialization bytes, if known.
    pub filtered_aggregate_estimated_materialization_bytes: Option<u64>,
    /// Runtime budget bytes consulted by aggregate execution, if known at optimization time.
    pub filtered_aggregate_memory_budget_bytes: Option<u64>,
    /// Runtime single-pass threshold bytes consulted by aggregate execution, if known at optimization time.
    pub filtered_aggregate_memory_threshold_bytes: Option<u64>,
    /// Stable threshold policy when exact runtime budget bytes are not optimizer-visible.
    pub(crate) filtered_aggregate_memory_threshold_policy: String,
    /// Abstract local-source fingerprints used for family grouping.
    pub(crate) source_fingerprints: Vec<String>,
    /// Number of exact source variants after retaining reader-predicate text.
    pub exact_source_variant_count: usize,
    /// Sum of selected source bytes across all repeated branches, if known.
    pub repeated_selected_source_bytes: Option<u64>,
    /// Conservative source-byte duplication estimate: total repeated bytes minus the largest branch.
    pub duplicate_selected_source_bytes: Option<u64>,
    /// True when the returned plan carries the shared materialization rewrite.
    pub applied: bool,
    /// Rejection reason when the candidate was discovered but not rewritten.
    pub rejected_reason: Option<SharedSubplanMaterializationRejectedReason>,
}

/// Reasons a shared-subplan materialization candidate was not rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedSubplanMaterializationRejectedReason {
    /// Branch source predicates could not be reconstructed after the aggregate.
    PredicateNotPostAggregateFilterable,
    /// The repeated scalar aggregate shape is not safe to combine into one filtered aggregate.
    UnsupportedScalarAggregateShape,
    /// Branch predicate columns could not be carried to the shared aggregate input.
    PredicateColumnPropagationUnsupported,
    /// The candidate was not reachable from the query sink.
    NotReachableFromSink,
}
