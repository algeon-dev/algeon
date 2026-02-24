//! Native plan preference facts and admission decisions.
//!
//! Capability analysis answers whether a plan can execute natively. Preference
//! analysis answers whether the current native profile should choose CPU for a
//! known high-overhead native shape. Decisions here must stay conservative:
//! missing evidence keeps GPU preferred.

use super::*;
use crate::source::LocalParquetSource;

/// Stable reason for preferring CPU execution over native GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativePlanCpuPreferenceReason {
    /// Small, selective source work feeds a bounded top-k shape whose native
    /// fixed source/join overhead is likely to dominate useful GPU work.
    SmallSelectiveTopK,
    /// Small, selective source work feeds a residual join, where native must
    /// pay residual materialization/evaluation overhead.
    SmallSelectiveResidualJoin,
}

impl NativePlanCpuPreferenceReason {
    /// Stable snake-case reason code.
    pub const fn code(self) -> &'static str {
        match self {
            Self::SmallSelectiveTopK => "native_small_selective_topk_cpu_preferred",
            Self::SmallSelectiveResidualJoin => "native_small_selective_residual_join_cpu_preferred",
        }
    }
}

/// Tunable conservative thresholds for native plan preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativePlanPreferenceOptions {
    /// Whether CPU preference rules are active.
    pub enabled: bool,
    /// Maximum selected local-Parquet row groups for a shape to be considered
    /// small/selective. All row-group evidence must be known.
    pub max_small_selective_row_groups: usize,
    /// Maximum selected source-read bytes for ordinary small/selective CPU
    /// preference. All local-Parquet source byte evidence must be known.
    pub max_small_selective_source_bytes: u64,
    /// Maximum selected local-Parquet row groups for residual-join CPU
    /// preference. Residual joins use a tighter threshold because CPU fallback
    /// often runs under the caller's native-session DataFusion config, not the
    /// tuned CPU benchmark config.
    pub max_small_selective_residual_join_row_groups: usize,
    /// Maximum selected source-read bytes for residual-join CPU preference.
    pub max_small_selective_residual_join_source_bytes: u64,
    /// Maximum `LIMIT` fetch that qualifies as a top-k/small-output boundary.
    pub max_topk_fetch: usize,
}

impl Default for NativePlanPreferenceOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_small_selective_row_groups: 128,
            max_small_selective_source_bytes: 1 << 30,
            max_small_selective_residual_join_row_groups: 16,
            max_small_selective_residual_join_source_bytes: 128 << 20,
            max_topk_fetch: 100,
        }
    }
}

impl NativePlanPreferenceOptions {
    /// Disables preference decisions while still allowing facts to be gathered.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            max_small_selective_row_groups: 128,
            max_small_selective_source_bytes: 1 << 30,
            max_small_selective_residual_join_row_groups: 16,
            max_small_selective_residual_join_source_bytes: 128 << 20,
            max_topk_fetch: 100,
        }
    }
}

/// Query-level facts used by preference admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanPreferenceFacts {
    /// Total native IR nodes.
    pub node_count: usize,
    /// Number of source nodes.
    pub source_count: usize,
    /// Number of local-Parquet source nodes.
    pub local_parquet_source_count: usize,
    /// True when every local-Parquet source has row-group metadata.
    pub selected_row_groups_known: bool,
    /// Total selected local-Parquet row groups when known.
    pub selected_row_groups: Option<usize>,
    /// Estimated source rows when all local-Parquet source rows are known.
    pub estimated_source_rows: Option<u64>,
    /// Estimated projected source bytes when all local-Parquet source bytes are known.
    pub estimated_source_bytes: Option<u64>,
    /// True when every local-Parquet source had selected row-group byte evidence.
    pub selected_source_bytes_known: bool,
    /// Number of join nodes.
    pub join_count: usize,
    /// Number of join nodes with residual predicates.
    pub residual_join_count: usize,
    /// Number of aggregate nodes.
    pub aggregate_count: usize,
    /// Number of grouped aggregate nodes.
    pub grouped_aggregate_count: usize,
    /// Number of sort nodes.
    pub sort_count: usize,
    /// Number of limit nodes.
    pub limit_count: usize,
    /// Minimum fetch for a `Sort -> Limit(fetch)` top-k boundary.
    pub min_topk_fetch: Option<usize>,
}

impl NativePlanPreferenceFacts {
    /// Builds preference facts from a native query plan.
    pub fn from_plan(plan: &QueryPlan) -> Self {
        let mut facts = Self {
            node_count: plan.node_count(),
            source_count: 0,
            local_parquet_source_count: 0,
            selected_row_groups_known: true,
            selected_row_groups: Some(0),
            estimated_source_rows: Some(0),
            estimated_source_bytes: Some(0),
            selected_source_bytes_known: true,
            join_count: 0,
            residual_join_count: 0,
            aggregate_count: 0,
            grouped_aggregate_count: 0,
            sort_count: 0,
            limit_count: 0,
            min_topk_fetch: None,
        };

        for node in plan.nodes() {
            match &node.kind {
                PlanNode::Source(source) => {
                    facts.source_count += 1;
                    if let SourceNode::LocalParquet(source) = source.as_ref() {
                        facts.local_parquet_source_count += 1;
                        accumulate_local_parquet_source_facts(&mut facts, source);
                    }
                },
                PlanNode::Join(join) => {
                    facts.join_count += 1;
                    if join.spec.residual_predicate.is_some() {
                        facts.residual_join_count += 1;
                    }
                },
                PlanNode::Aggregate(aggregate) => {
                    facts.aggregate_count += 1;
                    if !aggregate.spec.group_exprs.is_empty() {
                        facts.grouped_aggregate_count += 1;
                    }
                },
                PlanNode::Sort(_) => facts.sort_count += 1,
                PlanNode::Limit(limit) => {
                    facts.limit_count += 1;
                    if let Some(fetch) = limit.spec.fetch
                        && matches!(plan.node(limit.input).map(|node| &node.kind), Some(PlanNode::Sort(_)))
                    {
                        facts.min_topk_fetch = Some(facts.min_topk_fetch.map_or(fetch, |current| current.min(fetch)));
                    }
                },
                PlanNode::Projection(_)
                | PlanNode::Filter(_)
                | PlanNode::Window(_)
                | PlanNode::Union(_)
                | PlanNode::EdgeNormalize(_)
                | PlanNode::GraphAlgorithm(_) => {},
            }
        }

        if facts.local_parquet_source_count == 0 {
            facts.selected_row_groups_known = false;
            facts.selected_row_groups = None;
            facts.estimated_source_rows = None;
            facts.estimated_source_bytes = None;
            facts.selected_source_bytes_known = false;
        }

        facts
    }

    /// Renders stable key-value facts for adapter diagnostics and tests.
    pub fn diagnostic_facts(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("node_count", self.node_count.to_string()),
            ("source_count", self.source_count.to_string()),
            ("local_parquet_source_count", self.local_parquet_source_count.to_string()),
            ("selected_row_groups_known", self.selected_row_groups_known.to_string()),
            ("selected_row_groups", optional_usize(self.selected_row_groups)),
            ("estimated_source_rows", optional_u64(self.estimated_source_rows)),
            ("estimated_source_bytes", optional_u64(self.estimated_source_bytes)),
            ("selected_source_bytes_known", self.selected_source_bytes_known.to_string()),
            ("join_count", self.join_count.to_string()),
            ("residual_join_count", self.residual_join_count.to_string()),
            ("aggregate_count", self.aggregate_count.to_string()),
            ("grouped_aggregate_count", self.grouped_aggregate_count.to_string()),
            ("sort_count", self.sort_count.to_string()),
            ("limit_count", self.limit_count.to_string()),
            ("min_topk_fetch", optional_usize(self.min_topk_fetch)),
        ])
    }
}

/// Preference admission decision for a native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativePlanPreferenceDecision {
    /// Native GPU remains preferred.
    GpuPreferred {
        /// Facts used to make the decision.
        facts: NativePlanPreferenceFacts,
    },
    /// CPU is preferred for a supported native plan.
    CpuPreferred {
        /// Stable reason for the CPU preference.
        reason: NativePlanCpuPreferenceReason,
        /// Facts used to make the decision.
        facts: NativePlanPreferenceFacts,
    },
}

impl NativePlanPreferenceDecision {
    /// Returns true when this decision prefers CPU execution.
    pub fn is_cpu_preferred(&self) -> bool {
        matches!(self, Self::CpuPreferred { .. })
    }

    /// Returns the decision facts.
    pub fn facts(&self) -> &NativePlanPreferenceFacts {
        match self {
            Self::GpuPreferred { facts } | Self::CpuPreferred { facts, .. } => facts,
        }
    }
}

/// Decides whether a native plan should run on GPU or fall back to CPU.
pub fn native_plan_preference_decision(
    plan: &QueryPlan,
    options: &NativePlanPreferenceOptions,
) -> NativePlanPreferenceDecision {
    let facts = NativePlanPreferenceFacts::from_plan(plan);
    if !options.enabled {
        return NativePlanPreferenceDecision::GpuPreferred { facts };
    }
    let Some(selected_row_groups) = facts.selected_row_groups else {
        return NativePlanPreferenceDecision::GpuPreferred { facts };
    };
    let Some(selected_source_bytes) = facts.estimated_source_bytes else {
        return NativePlanPreferenceDecision::GpuPreferred { facts };
    };
    if !facts.selected_row_groups_known
        || !facts.selected_source_bytes_known
        || selected_row_groups > options.max_small_selective_row_groups
        || selected_source_bytes > options.max_small_selective_source_bytes
    {
        return NativePlanPreferenceDecision::GpuPreferred { facts };
    }
    if facts.residual_join_count > 0 {
        if selected_row_groups <= options.max_small_selective_residual_join_row_groups
            && selected_source_bytes <= options.max_small_selective_residual_join_source_bytes
        {
            return NativePlanPreferenceDecision::CpuPreferred {
                reason: NativePlanCpuPreferenceReason::SmallSelectiveResidualJoin,
                facts,
            };
        }
        return NativePlanPreferenceDecision::GpuPreferred { facts };
    }
    if facts.min_topk_fetch.is_some_and(|fetch| fetch <= options.max_topk_fetch)
        && (facts.join_count > 0 || facts.grouped_aggregate_count > 0)
    {
        return NativePlanPreferenceDecision::CpuPreferred {
            reason: NativePlanCpuPreferenceReason::SmallSelectiveTopK,
            facts,
        };
    }
    NativePlanPreferenceDecision::GpuPreferred { facts }
}

fn accumulate_local_parquet_source_facts(facts: &mut NativePlanPreferenceFacts, source: &LocalParquetSource) {
    let row_groups = source.selected_row_group_count();
    if row_groups == 0 {
        facts.selected_row_groups_known = false;
        facts.selected_row_groups = None;
    } else if let Some(total) = facts.selected_row_groups.as_mut() {
        *total = total.saturating_add(row_groups);
    }

    let rows = source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_rows);
    facts.estimated_source_rows = match (facts.estimated_source_rows, rows) {
        (Some(total), Some(rows)) => total.checked_add(rows),
        _ => None,
    };

    let bytes = source.diagnostics.selected_row_group_stats.as_ref().and_then(|stats| stats.selected_source_bytes());
    if bytes.is_none() {
        facts.selected_source_bytes_known = false;
    }
    facts.estimated_source_bytes = match (facts.estimated_source_bytes, bytes) {
        (Some(total), Some(bytes)) => total.checked_add(bytes),
        _ => None,
    };
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}
