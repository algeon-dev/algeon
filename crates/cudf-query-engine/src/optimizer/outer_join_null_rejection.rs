use crate::{
    expr::{ColumnRef, Expr},
    plan::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan},
};

pub const OUTER_JOIN_NULL_REJECTION_RULE_NAME: &str = "outer_join_null_rejection";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OuterJoinNullRejectionOptions {
    pub enabled: bool,
}

impl OuterJoinNullRejectionOptions {
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for OuterJoinNullRejectionOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OuterJoinNullRejectionDiagnostics {
    pub considered_count: usize,
    pub applied_count: usize,
    pub candidates: Vec<OuterJoinNullRejectionCandidateDiagnostics>,
}

impl OuterJoinNullRejectionDiagnostics {
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OuterJoinNullRejectionCandidateDiagnostics {
    pub rule_name: &'static str,
    pub candidate_node_id: NodeId,
    pub rejecting_node_id: NodeId,
    pub rejecting_join_side: JoinSide,
    pub rejecting_columns: Vec<String>,
    pub applied: bool,
    pub rejected_reason: Option<OuterJoinNullRejectionRejectedReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterJoinNullRejectionRejectedReason {
    UnsupportedRejectingJoinShape,
    NoRightSideNullRejectingKey,
    SharedOuterJoinOutput,
}

pub(crate) struct OuterJoinNullRejectionResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: OuterJoinNullRejectionDiagnostics,
}

pub(crate) fn optimize_outer_join_null_rejection(
    plan: QueryPlan,
    options: &OuterJoinNullRejectionOptions,
) -> OuterJoinNullRejectionResult {
    if !options.enabled {
        return OuterJoinNullRejectionResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: OuterJoinNullRejectionDiagnostics::default(),
        };
    }

    let consumer_counts = consumer_counts(&plan);
    let mut nodes = plan.nodes().to_vec();
    let mut diagnostics = OuterJoinNullRejectionDiagnostics::default();
    let mut changed = false;

    for parent_index in 0..nodes.len() {
        let parent_node = nodes[parent_index].clone();
        let PlanNode::Join(parent_join) = &parent_node.kind else {
            continue;
        };

        for rejecting_side in [JoinSide::Left, JoinSide::Right] {
            let mut candidate = OuterJoinNullRejectionCandidateDiagnostics {
                rule_name: OUTER_JOIN_NULL_REJECTION_RULE_NAME,
                candidate_node_id: NodeId::from_index_for_internal_use(0),
                rejecting_node_id: parent_node.id,
                rejecting_join_side: rejecting_side,
                rejecting_columns: Vec::new(),
                applied: false,
                rejected_reason: Some(OuterJoinNullRejectionRejectedReason::UnsupportedRejectingJoinShape),
            };

            if !rejecting_join_supported(parent_join) {
                continue;
            }

            let traces = right_side_null_rejecting_traces(&nodes, parent_join, rejecting_side);
            let Some(first_trace) = traces.first() else {
                continue;
            };

            diagnostics.considered_count += 1;
            candidate.candidate_node_id = first_trace.outer_join_id;
            candidate.rejecting_columns = traces.iter().map(|trace| trace.rejecting_column.clone()).collect();

            if !traces.iter().all(|trace| trace.outer_join_id == first_trace.outer_join_id) {
                candidate.rejected_reason = Some(OuterJoinNullRejectionRejectedReason::NoRightSideNullRejectingKey);
                diagnostics.candidates.push(candidate);
                continue;
            }

            if !path_is_exclusive(&consumer_counts, &first_trace.path) {
                candidate.rejected_reason = Some(OuterJoinNullRejectionRejectedReason::SharedOuterJoinOutput);
                diagnostics.candidates.push(candidate);
                continue;
            }

            if let Some(node) = nodes.get_mut(first_trace.outer_join_id.index())
                && let PlanNode::Join(join) = &mut node.kind
                && join.spec.join_type == JoinType::Left
            {
                join.spec.join_type = JoinType::Inner;
                candidate.applied = true;
                candidate.rejected_reason = None;
                diagnostics.applied_count += 1;
                changed = true;
            }
            diagnostics.candidates.push(candidate);
        }
    }

    let change = if changed { super::NativeOptimizerChange::Changed } else { super::NativeOptimizerChange::Unchanged };
    let sink = plan.sink().clone();
    let root = plan.root_id();
    OuterJoinNullRejectionResult { plan: QueryPlan::from_rewritten_parts(nodes, root, sink), change, diagnostics }
}

fn rejecting_join_supported(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && join.spec.null_equality == crate::plan::JoinNullEquality::NullsDoNotMatch
}

fn right_side_null_rejecting_traces(
    nodes: &[crate::plan::QueryNode],
    join: &JoinNode,
    side: JoinSide,
) -> Vec<NullRejectingTrace> {
    let input = join.inputs[join_side_index(side)];
    parent_key_columns(join, side)
        .iter()
        .filter_map(|column| {
            trace_right_origin(nodes, input, column.name()).map(|mut trace| {
                trace.rejecting_column = column.name().to_string();
                trace
            })
        })
        .collect()
}

fn parent_key_columns(join: &JoinNode, side: JoinSide) -> Vec<&ColumnRef> {
    join.spec
        .keys
        .iter()
        .map(|key| match side {
            JoinSide::Left => &key.left,
            JoinSide::Right => &key.right,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NullRejectingTrace {
    outer_join_id: NodeId,
    rejecting_column: String,
    path: Vec<NodeId>,
}

fn trace_right_origin(
    nodes: &[crate::plan::QueryNode],
    mut node_id: NodeId,
    column_name: &str,
) -> Option<NullRejectingTrace> {
    let mut current_column = column_name.to_string();
    let mut path = Vec::new();

    loop {
        path.push(node_id);
        let node = nodes.get(node_id.index()).filter(|node| node.id == node_id)?;
        match &node.kind {
            PlanNode::Projection(projection) => {
                let expr = unique_projection_expr_for_output(projection, &current_column)?;
                let column = simple_projection_column(expr)?;
                current_column = column.name().to_string();
                node_id = projection.input;
            },
            PlanNode::Filter(filter) => {
                node_id = filter.input;
            },
            PlanNode::Sort(sort) => {
                node_id = sort.input;
            },
            PlanNode::Limit(limit) => {
                node_id = limit.input;
            },
            PlanNode::Join(join) => {
                if join.spec.join_type != JoinType::Left {
                    return None;
                }
                let output = unique_join_output_for_name(join, &current_column)?;
                if output.side != JoinSide::Right {
                    return None;
                }
                return Some(NullRejectingTrace { outer_join_id: node_id, rejecting_column: String::new(), path });
            },
            PlanNode::Source(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => return None,
        }
    }
}

fn unique_projection_expr_for_output<'a>(
    projection: &'a crate::plan::ProjectionNode,
    output_name: &str,
) -> Option<&'a Expr> {
    let mut matches = projection.outputs.iter().filter(|output| output.field.name() == output_name);
    let first = matches.next()?;
    matches.next().is_none().then_some(&first.expr)
}

fn simple_projection_column(expr: &Expr) -> Option<&ColumnRef> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias { expr, .. } => simple_projection_column(expr),
        _ => None,
    }
}

fn unique_join_output_for_name<'a>(join: &'a JoinNode, output_name: &str) -> Option<&'a crate::plan::JoinOutputColumn> {
    let mut matches = join.spec.output_projection.iter().filter(|output| output.output_name == output_name);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn path_is_exclusive(consumer_counts: &[usize], path: &[NodeId]) -> bool {
    path.iter().all(|node_id| consumer_counts.get(node_id.index()).copied().unwrap_or_default() <= 1)
}

fn consumer_counts(plan: &QueryPlan) -> Vec<usize> {
    let mut counts = vec![0; plan.nodes().len()];
    for node in plan.nodes() {
        for input in node.input_ids() {
            if let Some(count) = counts.get_mut(input.index()) {
                *count += 1;
            }
        }
    }
    if let Some(count) = counts.get_mut(plan.sink().input().index()) {
        *count += 1;
    }
    counts
}

const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}
