use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};

use super::rewrite;
use crate::{
    expr::{ColumnRef, Expr},
    plan::{
        FilterNode, JoinNode, JoinOutputColumn, JoinSide, JoinType, NodeId, PlanFacts, PlanNode, ProjectionNode,
        QueryPlan, SchemaBehavior, estimated_output_bytes,
    },
};

/// Stable rule name used in optimizer diagnostics.
pub const OUTER_JOIN_NULL_REJECTION_RULE_NAME: &str = "outer_join_null_rejection";

const LARGE_NULL_REJECTED_RIGHT_INPUT_BYTES: u64 = 64 * 1024 * 1024;

/// Configuration for rewriting null-rejected outer joins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OuterJoinNullRejectionOptions {
    /// Whether the pass is allowed to rewrite eligible joins.
    pub enabled: bool,
}

impl OuterJoinNullRejectionOptions {
    /// Returns options with the pass disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for OuterJoinNullRejectionOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for outer-join null-rejection candidates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OuterJoinNullRejectionDiagnostics {
    /// Number of candidates considered.
    pub considered_count: usize,
    /// Number of candidates rewritten.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<OuterJoinNullRejectionCandidateDiagnostics>,
}

impl OuterJoinNullRejectionDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate evidence for outer-join null rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OuterJoinNullRejectionCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Outer join node that may be rewritten.
    pub candidate_node_id: NodeId,
    /// Downstream join whose keys reject nulls from the outer join.
    pub rejecting_node_id: NodeId,
    /// Side of the downstream join that rejects nulls.
    pub rejecting_join_side: JoinSide,
    /// Key columns proving null rejection.
    pub rejecting_columns: Vec<String>,
    /// True when the returned plan contains the rewrite.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<OuterJoinNullRejectionRejectedReason>,
}

/// Reasons an outer-join null-rejection candidate was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterJoinNullRejectionRejectedReason {
    /// Downstream join shape cannot prove null rejection.
    UnsupportedRejectingJoinShape,
    /// No right-side null-rejecting key reached the outer join.
    NoRightSideNullRejectingKey,
    /// Outer-join output is shared, so rewriting would affect another consumer.
    SharedOuterJoinOutput,
    /// The null-introducing right input is large enough that preserving the
    /// original outer-join execution shape is preferred.
    LargeNullIntroducingInput,
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

    for projection_index in 0..nodes.len() {
        let projection_node = nodes[projection_index].clone();
        let PlanNode::Projection(projection) = &projection_node.kind else {
            continue;
        };
        let filter_id = projection.input;
        let Some(filter_node) = nodes.get(filter_id.index()).filter(|node| node.id == filter_id).cloned() else {
            continue;
        };
        let PlanNode::Filter(filter) = &filter_node.kind else {
            continue;
        };
        let join_id = filter.input;
        let Some(join_node) = nodes.get(join_id.index()).filter(|node| node.id == join_id).cloned() else {
            continue;
        };
        let PlanNode::Join(join) = &join_node.kind else {
            continue;
        };
        if consumer_counts.get(join_id.index()).copied().unwrap_or_default() > 1
            || consumer_counts.get(filter_id.index()).copied().unwrap_or_default() > 1
        {
            continue;
        }
        let Some(rewrite) =
            left_join_marker_null_filter_to_anti_rewrite(join_node.output_schema(), join, filter, projection)
        else {
            continue;
        };

        diagnostics.considered_count += 1;
        diagnostics.applied_count += 1;
        diagnostics.candidates.push(OuterJoinNullRejectionCandidateDiagnostics {
            rule_name: OUTER_JOIN_NULL_REJECTION_RULE_NAME,
            candidate_node_id: join_id,
            rejecting_node_id: filter_id,
            rejecting_join_side: JoinSide::Right,
            rejecting_columns: vec![rewrite.marker_name.clone()],
            applied: true,
            rejected_reason: None,
        });

        nodes[join_id.index()].kind = PlanNode::Join(rewrite.join);
        nodes[join_id.index()].facts = PlanFacts::device(Arc::clone(&rewrite.output_schema), SchemaBehavior::Join);
        match rewrite.remaining_predicate {
            Some(predicate) => {
                nodes[filter_id.index()].kind = PlanNode::Filter(FilterNode { input: join_id, predicate });
                nodes[filter_id.index()].facts =
                    PlanFacts::device(Arc::clone(&rewrite.output_schema), SchemaBehavior::PreservesInput);
            },
            None => {
                nodes[projection_index].kind =
                    PlanNode::Projection(ProjectionNode::new(join_id, projection.outputs.clone()));
            },
        }
        changed = true;
    }

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

            if large_null_introducing_input_requires_preserved_shape(&plan, &nodes, first_trace.outer_join_id, &traces)
            {
                candidate.rejected_reason = Some(OuterJoinNullRejectionRejectedReason::LargeNullIntroducingInput);
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
    let root = plan.root_id();
    OuterJoinNullRejectionResult {
        plan: rewrite::finish_rewritten_plan(nodes, root, plan.sink())
            .with_keyed_hash_join_strategy_options_from(&plan),
        change,
        diagnostics,
    }
}

#[derive(Debug, Clone)]
struct LeftAntiMarkerRewrite {
    join: JoinNode,
    output_schema: SchemaRef,
    marker_name: String,
    remaining_predicate: Option<Expr>,
}

fn left_join_marker_null_filter_to_anti_rewrite(
    join_output_schema: &SchemaRef,
    join: &JoinNode,
    filter: &FilterNode,
    projection: &ProjectionNode,
) -> Option<LeftAntiMarkerRewrite> {
    if join.spec.join_type != JoinType::Left
        || join.spec.build_side != JoinSide::Right
        || join.spec.keys.is_empty()
        || join.spec.residual_predicate.is_some()
        || join.spec.null_equality != crate::plan::JoinNullEquality::NullsDoNotMatch
    {
        return None;
    }

    let right_outputs =
        join.spec.output_projection.iter().filter(|column| column.side == JoinSide::Right).collect::<Vec<_>>();
    if right_outputs.is_empty() {
        return None;
    }
    let right_output_names = right_outputs.iter().map(|column| column.output_name.as_str()).collect::<Vec<_>>();
    let (marker_name, remaining_predicate) =
        split_single_marker_is_null_predicate(&filter.predicate, &right_output_names)?;
    let marker_output = unique_right_marker_output(&right_outputs, &marker_name)?;
    if !right_marker_is_join_key(join, marker_output) {
        return None;
    }

    if projection.outputs.iter().any(|output| {
        right_output_names.contains(&output.field.name().as_str())
            || expr_references_any_column_name(&output.expr, &right_output_names)
    }) {
        return None;
    }
    if remaining_predicate.as_ref().is_some_and(|expr| expr_references_any_column_name(expr, &right_output_names)) {
        return None;
    }

    let output_projection =
        join.spec.output_projection.iter().filter(|column| column.side == JoinSide::Left).cloned().collect::<Vec<_>>();
    if output_projection.is_empty() {
        return None;
    }
    let output_schema = left_anti_output_schema(join_output_schema, &output_projection)?;
    let mut anti = join.clone();
    anti.spec.join_type = JoinType::LeftAnti;
    anti.spec.output_projection = output_projection;
    anti.spec.mark_column_name = None;
    Some(LeftAntiMarkerRewrite { join: anti, output_schema, marker_name, remaining_predicate })
}

fn unique_right_marker_output<'a>(
    right_outputs: &'a [&'a JoinOutputColumn],
    marker_name: &str,
) -> Option<&'a JoinOutputColumn> {
    let mut matches = right_outputs.iter().copied().filter(|output| {
        output.output_name == marker_name
            || output.column.name() == marker_name
            || output.column.display_name() == marker_name
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn right_marker_is_join_key(join: &JoinNode, marker_output: &JoinOutputColumn) -> bool {
    join.spec.keys.iter().any(|key| {
        marker_output.column.name() == key.right.name()
            || marker_output.column.display_name() == key.right.display_name()
            || marker_output.column.name() == key.right.display_name()
            || marker_output.column.display_name() == key.right.name()
    })
}

fn left_anti_output_schema(
    join_output_schema: &SchemaRef,
    output_projection: &[JoinOutputColumn],
) -> Option<SchemaRef> {
    let mut fields = Vec::with_capacity(output_projection.len());
    for output in output_projection {
        fields.push(join_output_schema.field_with_name(&output.output_name).ok()?.clone());
    }
    Some(Arc::new(Schema::new(fields)))
}

fn split_single_marker_is_null_predicate(expr: &Expr, marker_names: &[&str]) -> Option<(String, Option<Expr>)> {
    let mut terms = Vec::new();
    flatten_conjunction(expr, &mut terms);
    let mut marker_index = None;
    let mut marker_name = None;
    for (index, term) in terms.iter().enumerate() {
        if let Some(name) = marker_is_null_term(term, marker_names) {
            if marker_index.is_some() {
                return None;
            }
            marker_index = Some(index);
            marker_name = Some(name);
        }
    }
    let marker_index = marker_index?;
    let remaining_terms = terms
        .into_iter()
        .enumerate()
        .filter_map(|(index, term)| (index != marker_index).then_some(term.clone()))
        .collect::<Vec<_>>();
    Some((marker_name?, rebuild_conjunction(remaining_terms)))
}

fn flatten_conjunction(expr: &Expr, terms: &mut Vec<Expr>) {
    match expr {
        Expr::Binary { left, op: crate::expr::BinaryOp::And, right, .. } => {
            flatten_conjunction(left, terms);
            flatten_conjunction(right, terms);
        },
        Expr::Alias { expr, .. } => flatten_conjunction(expr, terms),
        _ => terms.push(expr.clone()),
    }
}

fn rebuild_conjunction(mut terms: Vec<Expr>) -> Option<Expr> {
    if terms.is_empty() {
        return None;
    }
    let first = terms.remove(0);
    Some(terms.into_iter().fold(first, |left, right| Expr::binary(left, crate::expr::BinaryOp::And, right)))
}

fn marker_is_null_term(expr: &Expr, marker_names: &[&str]) -> Option<String> {
    match expr {
        Expr::IsNull(inner) => column_matches_any_name(inner, marker_names),
        Expr::Alias { expr, .. } => marker_is_null_term(expr, marker_names),
        _ => None,
    }
}

fn column_matches_any_name(expr: &Expr, marker_names: &[&str]) -> Option<String> {
    let Expr::Column(column) = expr else {
        return None;
    };
    marker_names
        .iter()
        .find(|name| **name == column.name() || **name == column.display_name())
        .map(|name| (*name).to_string())
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
        .flat_map(|column| {
            trace_right_origins(nodes, input, column.name()).into_iter().map(|mut trace| {
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

fn trace_right_origins(
    nodes: &[crate::plan::QueryNode],
    node_id: NodeId,
    column_name: &str,
) -> Vec<NullRejectingTrace> {
    let mut traces = Vec::new();
    trace_right_origins_inner(nodes, node_id, column_name, &mut Vec::new(), &mut traces);
    traces
}

fn trace_right_origins_inner(
    nodes: &[crate::plan::QueryNode],
    node_id: NodeId,
    column_name: &str,
    path: &mut Vec<NodeId>,
    traces: &mut Vec<NullRejectingTrace>,
) {
    if path.contains(&node_id) {
        return;
    }
    let Some(node) = nodes.get(node_id.index()).filter(|node| node.id == node_id) else {
        return;
    };

    path.push(node_id);
    match &node.kind {
        PlanNode::Projection(projection) => {
            if let Some(expr) = unique_projection_expr_for_output(projection, column_name)
                && let Some(column) = simple_projection_column(expr)
            {
                trace_right_origins_inner(nodes, projection.input, column.name(), path, traces);
            }
        },
        PlanNode::Filter(filter) => {
            trace_right_origins_inner(nodes, filter.input, column_name, path, traces);
        },
        PlanNode::Aggregate(aggregate) => {
            if let Some(column) = aggregate_group_input_column_for_output(node, aggregate, column_name) {
                trace_right_origins_inner(nodes, aggregate.input, column.name(), path, traces);
            }
        },
        PlanNode::Sort(sort) => {
            trace_right_origins_inner(nodes, sort.input, column_name, path, traces);
        },
        // Limit is not transparent to this rewrite: null-rejected rows can
        // consume limit slots before a downstream join removes them.
        PlanNode::Limit(_) => {},
        PlanNode::Join(join) => {
            if let Some(output) = unique_join_output_for_name(join, column_name) {
                match join.spec.join_type {
                    JoinType::Left if output.side == JoinSide::Right => {
                        traces.push(NullRejectingTrace {
                            outer_join_id: node_id,
                            rejecting_column: String::new(),
                            path: path.clone(),
                        });
                    },
                    JoinType::Inner => {
                        let input = join.inputs[join_side_index(output.side)];
                        trace_right_origins_inner(nodes, input, output.column.name(), path, traces);
                    },
                    _ => {},
                }
            }
        },
        PlanNode::Union(union) => {
            for input in &union.inputs {
                trace_right_origins_inner(nodes, *input, column_name, path, traces);
            }
        },
        PlanNode::Source(_) | PlanNode::Window(_) | PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) => {},
    }
    path.pop();
}

fn aggregate_group_input_column_for_output<'a>(
    node: &'a crate::plan::QueryNode,
    aggregate: &'a crate::plan::AggregateNode,
    output_name: &str,
) -> Option<&'a ColumnRef> {
    if aggregate.spec.grouping_sets.is_some() {
        return None;
    }

    for (index, expr) in aggregate.spec.group_exprs.iter().enumerate() {
        let field = node.output_schema().fields().get(index)?;
        if field.name() == output_name {
            return simple_projection_column(expr);
        }
    }
    None
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

fn expr_references_any_column_name(expr: &Expr, column_names: &[&str]) -> bool {
    match expr {
        Expr::Column(column) => column_names.contains(&column.name()) || column_names.contains(&column.display_name()),
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. } => expr_references_any_column_name(expr, column_names),
        Expr::Binary { left, right, .. } => {
            expr_references_any_column_name(left, column_names) || expr_references_any_column_name(right, column_names)
        },
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            expr_references_any_column_name(left, column_names) || expr_references_any_column_name(right, column_names)
        },
        Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. } => expr_references_any_column_name(expr, column_names),
        Expr::Concat(exprs) => exprs.iter().any(|expr| expr_references_any_column_name(expr, column_names)),
        Expr::Case { branches, else_expr } => {
            branches.iter().any(|branch| {
                expr_references_any_column_name(&branch.when, column_names)
                    || expr_references_any_column_name(&branch.then, column_names)
            }) || expr_references_any_column_name(else_expr, column_names)
        },
        Expr::InList { expr, list, .. } => {
            expr_references_any_column_name(expr, column_names)
                || list.iter().any(|expr| expr_references_any_column_name(expr, column_names))
        },
        Expr::Literal(_) => false,
    }
}

fn path_is_exclusive(consumer_counts: &[usize], path: &[NodeId]) -> bool {
    path.iter().all(|node_id| consumer_counts.get(node_id.index()).copied().unwrap_or_default() <= 1)
}

fn large_null_introducing_input_requires_preserved_shape(
    plan: &QueryPlan,
    nodes: &[crate::plan::QueryNode],
    outer_join_id: NodeId,
    traces: &[NullRejectingTrace],
) -> bool {
    // Keep the large-input guard for direct consumers. When aggregate pushdown
    // has carried the rejected right key as a group key, null-extended rows
    // can only form groups that the downstream inner join discards.
    null_introducing_input_is_large(plan, outer_join_id) && !traces_cross_group_key_aggregate(nodes, traces)
}

fn traces_cross_group_key_aggregate(nodes: &[crate::plan::QueryNode], traces: &[NullRejectingTrace]) -> bool {
    traces.iter().any(|trace| {
        trace.path.iter().any(|node_id| {
            nodes
                .get(node_id.index())
                .filter(|node| node.id == *node_id)
                .is_some_and(|node| matches!(&node.kind, PlanNode::Aggregate(_)))
        })
    })
}

fn null_introducing_input_is_large(plan: &QueryPlan, outer_join_id: NodeId) -> bool {
    let Some(node) = plan.node(outer_join_id) else {
        return false;
    };
    let PlanNode::Join(join) = &node.kind else {
        return false;
    };
    let right_input = join.inputs[join_side_index(JoinSide::Right)];
    estimated_output_bytes(plan, right_input).is_some_and(|bytes| bytes > LARGE_NULL_REJECTED_RIGHT_INPUT_BYTES)
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
