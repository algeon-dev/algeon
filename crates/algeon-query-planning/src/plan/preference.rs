//! Native plan preference rules.
//!
//! Capability analysis says whether a plan can run natively. The preference
//! rules here keep a native-capable plan on the frontend when native execution
//! would hold a GPU grant without computing on column data.

use super::{PlanNode, QueryPlan, ResultSink, SourceNode};

/// Stable reason for not selecting a native-capable plan by cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativePlanCostReason {
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
            Self::NoColumnData => "native_no_column_data_not_selected_by_cost",
            Self::HostLiteralPassthrough => "native_host_literal_passthrough_not_selected_by_cost",
        }
    }
}

/// Returns the preference rule that keeps a native-capable plan on the
/// frontend, or `None` when native execution stays selected.
///
/// `frontend_plan_allowed` says whether the frontend may run its own plan in
/// place of the native candidate. When it may not, no rule applies. Otherwise
/// a plan that reads no column data, or that passes host literal rows back to
/// the host, stays with the frontend.
pub fn native_plan_cost_decision(plan: &QueryPlan, frontend_plan_allowed: bool) -> Option<NativePlanCostReason> {
    if !frontend_plan_allowed {
        return None;
    }
    if reads_no_column_data(plan) {
        return Some(NativePlanCostReason::NoColumnData);
    }
    if passes_host_literals_through(plan) {
        return Some(NativePlanCostReason::HostLiteralPassthrough);
    }
    None
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
