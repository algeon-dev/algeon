//! Native plan cost facts and selection decisions.
//!
//! Capability analysis answers whether a plan can execute natively. Cost
//! analysis answers whether the current native profile should leave a known
//! high-overhead native shape with DataFusion. Decisions here must stay
//! conservative: missing evidence continues native selection.

use std::collections::BTreeMap;

use super::{PlanNode, QueryPlan, ResultSink, SourceNode};
use crate::source::LocalParquetSource;

/// Stable reason for not selecting a native-capable plan by cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativePlanCostReason {
    /// Small, selective source work feeds a bounded top-k shape whose native
    /// fixed source/join overhead is likely to dominate useful GPU work.
    SmallSelectiveTopK,
    /// Small, selective source work feeds a residual join, where native must
    /// pay residual materialization/evaluation overhead.
    SmallSelectiveResidualJoin,
    /// Small, selective source work feeds a full-sort join/aggregate shape whose
    /// native source/join/sort fixed costs are likely to dominate useful GPU work.
    SmallSelectiveFullSort,
    /// Every source is a literal without columns, such as a DDL result's
    /// empty relation or a `SELECT 1` placeholder row, and no GPU algorithm
    /// runs: native execution would hold a grant to produce rows no column
    /// data distinguishes.
    NoColumnData,
    /// Every node is a literal source of host rows and the result returns to
    /// the host, such as a scan of an in-memory table under a DataFusion
    /// operator: native execution would copy the rows to the device and back
    /// without computing on them, and hold a grant to do so.
    HostLiteralPassthrough,
}

impl NativePlanCostReason {
    /// Stable snake-case reason code.
    pub const fn code(self) -> &'static str {
        match self {
            Self::SmallSelectiveTopK => "native_small_selective_topk_not_selected_by_cost",
            Self::SmallSelectiveResidualJoin => "native_small_selective_residual_join_not_selected_by_cost",
            Self::SmallSelectiveFullSort => "native_small_selective_full_sort_not_selected_by_cost",
            Self::NoColumnData => "native_no_column_data_not_selected_by_cost",
            Self::HostLiteralPassthrough => "native_host_literal_passthrough_not_selected_by_cost",
        }
    }
}

/// Tunable conservative thresholds for native plan cost selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativePlanCostOptions {
    /// Whether cost selection rules are active.
    pub enabled: bool,
    /// Maximum selected local-Parquet row groups for a shape to be considered
    /// small/selective. All row-group evidence must be known.
    pub max_small_selective_row_groups: usize,
    /// Maximum selected source-read bytes for ordinary small/selective cost
    /// selection. All local-Parquet source byte evidence must be known.
    pub max_small_selective_source_bytes: u64,
    /// Maximum selected local-Parquet row groups for residual-join cost
    /// selection. Residual joins use a tighter threshold because their
    /// DataFusion comparison often runs under a different session config than
    /// the native benchmark.
    pub max_small_selective_residual_join_row_groups: usize,
    /// Maximum selected source-read bytes for residual-join cost selection.
    pub max_small_selective_residual_join_source_bytes: u64,
    /// Maximum `LIMIT` fetch that qualifies as a top-k/small-output boundary.
    pub max_topk_fetch: usize,
}

impl Default for NativePlanCostOptions {
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

impl NativePlanCostOptions {
    /// Disables cost decisions while still allowing facts to be gathered.
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

/// Query-level facts used by cost selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanCostFacts {
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

impl NativePlanCostFacts {
    /// Builds cost facts from a native query plan.
    pub(crate) fn from_plan(plan: &QueryPlan) -> Self {
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
                    if let Some(fetch) = limit.spec.fetch {
                        let is_topk = match plan.node(limit.input).map(|node| &node.kind) {
                            Some(PlanNode::Sort(_)) => true,
                            Some(PlanNode::Projection(projection)) => {
                                matches!(plan.node(projection.input).map(|node| &node.kind), Some(PlanNode::Sort(_)))
                            },
                            _ => false,
                        };
                        if is_topk {
                            facts.min_topk_fetch =
                                Some(facts.min_topk_fetch.map_or(fetch, |current| current.min(fetch)));
                        }
                    }
                },
                PlanNode::Projection(_)
                | PlanNode::Filter(_)
                | PlanNode::Window(_)
                | PlanNode::Union(_)
                | PlanNode::EdgeNormalize(_)
                | PlanNode::GraphAlgorithm(_)
                | PlanNode::VectorAlgorithm(_)
                | PlanNode::MlAlgorithm(_) => {},
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

/// Cost-selection decision for a native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativePlanCostDecision {
    /// Native execution remains selected.
    ContinueNative {
        /// Facts used to make the decision.
        facts: NativePlanCostFacts,
    },
    /// A native-capable plan was not selected by cost.
    NotSelectedByCost {
        /// Stable cost-selection reason.
        reason: NativePlanCostReason,
        /// Facts used to make the decision.
        facts: NativePlanCostFacts,
    },
}

impl NativePlanCostDecision {
    /// Returns true when native execution was not selected by cost.
    pub fn is_not_selected_by_cost(&self) -> bool {
        matches!(self, Self::NotSelectedByCost { .. })
    }

    /// Returns the decision facts.
    pub fn facts(&self) -> &NativePlanCostFacts {
        match self {
            Self::ContinueNative { facts } | Self::NotSelectedByCost { facts, .. } => facts,
        }
    }
}

/// Decides whether a native-capable plan should be selected by cost.
///
/// `frontend_plan_allowed` says whether the frontend may run its own plan in
/// place of the native candidate. A plan that reads no column data stays with
/// the frontend whenever it may; the threshold rules in `options` apply only
/// when they are enabled.
pub fn native_plan_cost_decision(
    plan: &QueryPlan,
    options: &NativePlanCostOptions,
    frontend_plan_allowed: bool,
) -> NativePlanCostDecision {
    let facts = NativePlanCostFacts::from_plan(plan);
    if frontend_plan_allowed && reads_no_column_data(plan) {
        return NativePlanCostDecision::NotSelectedByCost { reason: NativePlanCostReason::NoColumnData, facts };
    }
    if frontend_plan_allowed && passes_host_literals_through(plan) {
        return NativePlanCostDecision::NotSelectedByCost {
            reason: NativePlanCostReason::HostLiteralPassthrough,
            facts,
        };
    }
    if !options.enabled {
        return NativePlanCostDecision::ContinueNative { facts };
    }
    let Some(selected_row_groups) = facts.selected_row_groups else {
        return NativePlanCostDecision::ContinueNative { facts };
    };
    let Some(selected_source_bytes) = facts.estimated_source_bytes else {
        return NativePlanCostDecision::ContinueNative { facts };
    };
    if !facts.selected_row_groups_known
        || !facts.selected_source_bytes_known
        || selected_row_groups > options.max_small_selective_row_groups
        || selected_source_bytes > options.max_small_selective_source_bytes
    {
        return NativePlanCostDecision::ContinueNative { facts };
    }
    if facts.residual_join_count > 0 {
        if selected_row_groups <= options.max_small_selective_residual_join_row_groups
            && selected_source_bytes <= options.max_small_selective_residual_join_source_bytes
        {
            return NativePlanCostDecision::NotSelectedByCost {
                reason: NativePlanCostReason::SmallSelectiveResidualJoin,
                facts,
            };
        }
        return NativePlanCostDecision::ContinueNative { facts };
    }
    if facts.min_topk_fetch.is_some_and(|fetch| fetch <= options.max_topk_fetch)
        && (facts.join_count > 0 || facts.grouped_aggregate_count > 0)
    {
        return NativePlanCostDecision::NotSelectedByCost { reason: NativePlanCostReason::SmallSelectiveTopK, facts };
    }
    if facts.sort_count > 0 && (facts.join_count > 0 || facts.grouped_aggregate_count > 0) {
        return NativePlanCostDecision::NotSelectedByCost {
            reason: NativePlanCostReason::SmallSelectiveFullSort,
            facts,
        };
    }
    NativePlanCostDecision::ContinueNative { facts }
}

/// Whether every source is a literal without column values: no columns, or no
/// rows (a relation DataFusion folded to empty, such as `WHERE false` or
/// `LIMIT 0`), and no GPU algorithm node runs.
fn reads_no_column_data(plan: &QueryPlan) -> bool {
    plan.nodes().iter().all(|node| match &node.kind {
        PlanNode::Source(source) => matches!(
            source.as_ref(),
            SourceNode::LiteralBatch(literal)
                if literal.schema.fields().is_empty() || literal.batches.iter().all(|batch| batch.num_rows() == 0)
        ),
        PlanNode::GraphAlgorithm(_) | PlanNode::VectorAlgorithm(_) | PlanNode::MlAlgorithm(_) => false,
        _ => true,
    })
}

/// Whether the plan returns to the host the literal rows it holds: every node
/// is a literal source and the result is not GPU-resident.
fn passes_host_literals_through(plan: &QueryPlan) -> bool {
    !matches!(plan.sink(), ResultSink::GpuResident { .. })
        && plan.nodes().iter().all(|node| {
            matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LiteralBatch(_)))
        })
}

fn accumulate_local_parquet_source_facts(facts: &mut NativePlanCostFacts, source: &LocalParquetSource) {
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
