//! Join-filter reassociation and residual-derived filter pushdown.
//!
//! Two rewrites share this rule and its diagnostics. Reassociation matches an
//! inner equi-join `P` whose output projects one side only (the preserved
//! side), whose preserved input is an inner join `C`, possibly under a renaming
//! projection, and whose preserved-side keys all trace to one child `a` of `C`.
//! It rewrites `P(C(a, b), f)` into `C(F(a, f), b)`, where `F` is a new inner
//! join over the same key values that keeps `a`'s columns, then re-applies
//! `P`'s projection. Row multiplicity is preserved because `f` matches the
//! same key values it matched before, `P`'s output carries no column of `f`,
//! and both joins require `NullsDoNotMatch`, so no null key changes sides.
//! Residual pushdown matches an inner join with a residual predicate and
//! inserts, below each input, the filter implied by every branch of the
//! residual for that side. The residual stays on the join, so the pushed
//! filter removes only rows the residual would remove.
//!
//! Both rewrites rebuild the node list, so `QueryPlan::rewrite` drops
//! dynamic-filter routes through changed nodes and invalidates advisory
//! bindings there; the rebuilt crossed join also discards its advisory byte
//! statistics because its columns changed. The optimizer driver runs the pass
//! to a fixpoint because one applied move can expose the next, and a pushed
//! filter is skipped when its exact predicate already heads the subtree, which
//! is what lets the fixpoint terminate. Rejection reasons are report contracts.

use std::{collections::BTreeMap, sync::Arc};

use arrow::datatypes::{DataType as ArrowDataType, SchemaRef};

use super::{
    dimension_pruning, rewrite, simple_projection_column, source_target_policy::OptimizerSourceTargetPolicy,
    unique_join_output_for_name,
};
use crate::{
    error::Result,
    expr::{BinaryOp, CaseWhen, ColumnQualifier, ColumnRef, Expr, LiteralValue},
    plan::{
        AdvisoryOutputStatistics, AdvisoryStatistic, FilterNode, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn,
        JoinSide, JoinSpec, JoinType, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode,
        QueryPlan, RewrittenNodes, estimated_output_bytes, estimated_output_rows,
    },
    source::SourceNode,
};

/// Stable rule name used in optimizer diagnostics.
pub const JOIN_FILTER_REASSOCIATION_RULE_NAME: &str = "join_filter_reassociation";

/// Configuration for moving filters across preserved-side join structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinFilterReassociationOptions {
    /// Whether the pass is allowed to rewrite eligible join/filter shapes.
    pub enabled: bool,
    /// Whether residual-derived necessary predicates may be pushed into join inputs.
    ///
    /// The rewrite keeps the original residual predicate in place and only
    /// pushes predicates that are implied by every surviving residual branch for
    /// a side. Large local-Parquet targets remain gated out unless the predicate
    /// can be remapped through transparent structure to a supported smaller
    /// target.
    pub residual_filter_pushdown_enabled: bool,
}

impl JoinFilterReassociationOptions {
    /// Returns options with the pass disabled.
    pub fn disabled() -> Self {
        Self { enabled: false, residual_filter_pushdown_enabled: false }
    }

    /// Enables residual-derived necessary-predicate pushdown for experiments
    /// and targeted tests.
    #[cfg(test)]
    pub fn with_residual_filter_pushdown(mut self, enabled: bool) -> Self {
        self.residual_filter_pushdown_enabled = enabled;
        self
    }
}

impl Default for JoinFilterReassociationOptions {
    fn default() -> Self {
        Self { enabled: true, residual_filter_pushdown_enabled: true }
    }
}

/// Diagnostics for join-filter reassociation candidates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinFilterReassociationDiagnostics {
    /// Number of candidates considered.
    pub considered_count: usize,
    /// Number of candidates rewritten.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<JoinFilterReassociationCandidateDiagnostics>,
}

impl JoinFilterReassociationDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub(crate) fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate evidence for join-filter reassociation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinFilterReassociationCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub(crate) rule_name: &'static str,
    /// Parent join candidate considered for reassociation.
    pub(crate) candidate_node_id: NodeId,
    /// Join crossed while moving the filter, if found.
    pub(crate) crossed_join_node_id: Option<NodeId>,
    /// Child subtree that receives the filter after reassociation.
    pub(crate) filtered_child_node_id: Option<NodeId>,
    /// Filter node introduced or moved by the rewrite.
    pub(crate) filtering_node_id: Option<NodeId>,
    /// True when the returned plan contains the rewrite.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<JoinFilterReassociationRejectedReason>,
}

/// Reasons a join-filter reassociation candidate was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinFilterReassociationRejectedReason {
    /// Parent join shape is not supported by the pass.
    UnsupportedParentJoinShape,
    /// Projection on the preserved side is not transparent enough to cross.
    NonPreservedSideProjection,
    /// Preserved input shape cannot safely receive the filter.
    UnsupportedPreservedInputShape,
    /// Required key lineage could not be found.
    MissingKeyLineage,
    /// Key lineage matched more than one possible target.
    AmbiguousKeyLineage,
    /// Crossed join shape is not safe for reassociation.
    UnsupportedCrossedJoinShape,
    /// Join residual did not contain a side-local necessary predicate.
    NoPushableResidualPredicate,
    /// Projected-parent reassociation would rewrite a projection shared by multiple consumers.
    SharedProjectedParent,
}

pub(crate) struct JoinFilterReassociationResult {
    pub(crate) plan: QueryPlan,
    pub(crate) change: super::NativeOptimizerChange,
    pub(crate) diagnostics: JoinFilterReassociationDiagnostics,
}

pub(crate) fn optimize_join_filter_reassociation(
    plan: QueryPlan,
    options: &JoinFilterReassociationOptions,
    resolver: &dyn dimension_pruning::LocalParquetPruningResolver,
) -> Result<JoinFilterReassociationResult> {
    if !options.enabled {
        return Ok(JoinFilterReassociationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: JoinFilterReassociationDiagnostics::default(),
        });
    }

    let mut rewriter = JoinFilterReassociationRewriter::new(&plan, options, resolver);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let diagnostics = rewriter.diagnostics;
        return Ok(JoinFilterReassociationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics,
        });
    }
    Ok(JoinFilterReassociationResult {
        plan: plan.rewrite(RewrittenNodes::rebuilt(rewriter.nodes, new_root, rewriter.memo))?,
        change: super::NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    })
}

/// Rebuilds the plan top down; `memo` maps each visited source node to its
/// image and becomes the origin map for `RewrittenNodes::rebuilt`.
struct JoinFilterReassociationRewriter<'a> {
    original: &'a QueryPlan,
    options: &'a JoinFilterReassociationOptions,
    /// Supplies Parquet footer statistics for scalar range derivation.
    resolver: &'a dyn dimension_pruning::LocalParquetPruningResolver,
    /// Parent counts with the sink counted as a consumer.
    consumer_counts: Vec<usize>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    diagnostics: JoinFilterReassociationDiagnostics,
    changed: bool,
}

impl<'a> JoinFilterReassociationRewriter<'a> {
    fn new(
        original: &'a QueryPlan,
        options: &'a JoinFilterReassociationOptions,
        resolver: &'a dyn dimension_pruning::LocalParquetPruningResolver,
    ) -> Self {
        Self {
            original,
            options,
            resolver,
            consumer_counts: node_consumer_counts(original),
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            diagnostics: JoinFilterReassociationDiagnostics::default(),
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        if self.options.residual_filter_pushdown_enabled
            && let PlanNode::Join(join) = &old_node.kind
            && let Some(new_id) = self.try_push_residual_filters_candidate(old_id, join)
        {
            self.memo.insert(old_id, new_id);
            return new_id;
        }
        if let PlanNode::Join(join) = &old_node.kind
            && let Some(new_id) = self.try_reassociate_candidate(old_id, join)
        {
            self.memo.insert(old_id, new_id);
            return new_id;
        }
        if let PlanNode::Projection(projection) = &old_node.kind
            && let Some(new_id) = self.try_reassociate_projected_join_candidate(old_id, projection)
        {
            self.memo.insert(old_id, new_id);
            return new_id;
        }

        let new_kind = self.clone_kind_with_rewritten_inputs(&old_node.kind);
        let new_id = self.push_node(new_kind, old_node.facts.clone());
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    fn try_reassociate_candidate(&mut self, candidate_node_id: NodeId, join: &JoinNode) -> Option<NodeId> {
        if !parent_join_supported(join) || parent_preserved_output_side(join).is_none() {
            return None;
        }

        let mut candidate = JoinFilterReassociationCandidateDiagnostics {
            rule_name: JOIN_FILTER_REASSOCIATION_RULE_NAME,
            candidate_node_id,
            crossed_join_node_id: None,
            filtered_child_node_id: None,
            filtering_node_id: None,
            applied: false,
            rejected_reason: None,
        };
        self.diagnostics.considered_count += 1;

        let output_schema =
            Arc::clone(self.original.node(candidate_node_id).expect("candidate node must exist").output_schema());
        match self.plan_reassociation(join, output_schema) {
            Ok(planned) => {
                candidate.crossed_join_node_id = Some(planned.crossed_join_node_id);
                candidate.filtered_child_node_id = Some(planned.filtered_child_node_id);
                candidate.filtering_node_id = Some(planned.filtering_node_id);
                candidate.applied = true;
                self.diagnostics.applied_count += 1;
                self.diagnostics.candidates.push(candidate);
                self.changed = true;
                Some(planned.new_id)
            },
            Err(reason) => {
                candidate.rejected_reason = Some(reason);
                self.diagnostics.candidates.push(candidate);
                None
            },
        }
    }

    fn try_reassociate_projected_join_candidate(
        &mut self,
        projection_node_id: NodeId,
        projection: &ProjectionNode,
    ) -> Option<NodeId> {
        let input_node = self.original.node(projection.input)?;
        let PlanNode::Join(join) = &input_node.kind else {
            return None;
        };
        if !parent_join_supported(join) {
            return None;
        }
        let projected_join = projected_preserved_side_join(join, projection)?;
        parent_preserved_output_side(&projected_join)?;

        let mut candidate = JoinFilterReassociationCandidateDiagnostics {
            rule_name: JOIN_FILTER_REASSOCIATION_RULE_NAME,
            candidate_node_id: projection.input,
            crossed_join_node_id: None,
            filtered_child_node_id: None,
            filtering_node_id: None,
            applied: false,
            rejected_reason: None,
        };
        self.diagnostics.considered_count += 1;
        if self.consumer_count(projection_node_id) > 1 {
            candidate.rejected_reason = Some(JoinFilterReassociationRejectedReason::SharedProjectedParent);
            self.diagnostics.candidates.push(candidate);
            return None;
        }

        match self.plan_reassociation(&projected_join, Arc::clone(projection.output_schema())) {
            Ok(planned) => {
                candidate.crossed_join_node_id = Some(planned.crossed_join_node_id);
                candidate.filtered_child_node_id = Some(planned.filtered_child_node_id);
                candidate.filtering_node_id = Some(planned.filtering_node_id);
                candidate.applied = true;
                self.diagnostics.applied_count += 1;
                self.diagnostics.candidates.push(candidate);
                self.changed = true;
                Some(planned.new_id)
            },
            Err(reason) => {
                candidate.rejected_reason = Some(reason);
                self.diagnostics.candidates.push(candidate);
                None
            },
        }
    }

    fn try_push_residual_filters_candidate(&mut self, candidate_node_id: NodeId, join: &JoinNode) -> Option<NodeId> {
        if !residual_pushdown_join_supported(join) {
            return None;
        }

        let mut candidate = JoinFilterReassociationCandidateDiagnostics {
            rule_name: JOIN_FILTER_REASSOCIATION_RULE_NAME,
            candidate_node_id,
            crossed_join_node_id: None,
            filtered_child_node_id: None,
            filtering_node_id: None,
            applied: false,
            rejected_reason: None,
        };
        self.diagnostics.considered_count += 1;

        match self.plan_residual_filter_pushdown(candidate_node_id, join) {
            Ok(planned) => {
                candidate.filtered_child_node_id = planned.filtered_child_node_id;
                candidate.filtering_node_id = planned.filtering_node_id;
                candidate.applied = true;
                self.diagnostics.applied_count += 1;
                self.diagnostics.candidates.push(candidate);
                self.changed = true;
                Some(planned.new_id)
            },
            Err(reason) => {
                candidate.rejected_reason = Some(reason);
                self.diagnostics.candidates.push(candidate);
                None
            },
        }
    }

    fn plan_residual_filter_pushdown(
        &mut self,
        candidate_node_id: NodeId,
        join: &JoinNode,
    ) -> std::result::Result<ResidualFilterPushdownPlan, JoinFilterReassociationRejectedReason> {
        let residual = join
            .spec
            .residual_predicate
            .as_ref()
            .ok_or(JoinFilterReassociationRejectedReason::NoPushableResidualPredicate)?;
        let left_filter = derive_side_necessary_filter(self.original, join, residual, JoinSide::Left, self.resolver);
        let right_filter = derive_side_necessary_filter(self.original, join, residual, JoinSide::Right, self.resolver);
        if left_filter.is_none() && right_filter.is_none() {
            return Err(JoinFilterReassociationRejectedReason::NoPushableResidualPredicate);
        }

        let mut new_join = join.clone();
        let left_plan = self.push_filter_if_new(join.inputs[0], left_filter);
        let right_plan = self.push_filter_if_new(join.inputs[1], right_filter);
        if left_plan.filtering_node_id.is_none() && right_plan.filtering_node_id.is_none() {
            return Err(JoinFilterReassociationRejectedReason::NoPushableResidualPredicate);
        }

        new_join.inputs = [left_plan.node_id, right_plan.node_id];
        let old_node = self.original.node(candidate_node_id).expect("optimizer input node must exist");
        let new_id = self.push_node(PlanNode::Join(new_join), old_node.facts.clone());
        Ok(ResidualFilterPushdownPlan {
            new_id,
            filtered_child_node_id: left_plan.filtered_child_node_id.or(right_plan.filtered_child_node_id),
            filtering_node_id: left_plan.filtering_node_id.or(right_plan.filtering_node_id),
        })
    }

    fn push_filter_if_new(&mut self, old_input: NodeId, predicate: Option<Expr>) -> PushedFilterInput {
        let Some(predicate) = predicate else {
            let input_new = self.rewrite_subtree(old_input);
            return PushedFilterInput { node_id: input_new, filtered_child_node_id: None, filtering_node_id: None };
        };
        self.push_filter_into_subtree(old_input, predicate)
    }

    /// Places `predicate` as deep as the structure allows: through filters,
    /// through projections that rename its columns, and through the one side
    /// of an inner join that owns all its columns, then wraps the subtree it
    /// reaches. A subtree that already starts with this exact predicate is
    /// left alone, which is why the fixpoint terminates.
    fn push_filter_into_subtree(&mut self, old_input: NodeId, predicate: Expr) -> PushedFilterInput {
        if input_has_top_filter(self.original, old_input, &predicate) {
            let input_new = self.rewrite_subtree(old_input);
            return PushedFilterInput { node_id: input_new, filtered_child_node_id: None, filtering_node_id: None };
        }

        let old_node = self.original.node(old_input).expect("optimizer input node must exist");
        match &old_node.kind {
            PlanNode::Filter(filter) => {
                let pushed = self.push_filter_into_subtree(filter.input, predicate);
                if pushed.filtering_node_id.is_none() {
                    let input_new = self.rewrite_subtree(old_input);
                    return PushedFilterInput {
                        node_id: input_new,
                        filtered_child_node_id: None,
                        filtering_node_id: None,
                    };
                }
                let mut rebuilt = filter.clone();
                rebuilt.input = pushed.node_id;
                let new_id = self.push_node(PlanNode::Filter(rebuilt), old_node.facts.clone());
                return PushedFilterInput { node_id: new_id, ..pushed };
            },
            PlanNode::Projection(projection) => {
                if let Some(remapped) = remap_projection_predicate(&predicate, projection) {
                    let pushed = self.push_filter_into_subtree(projection.input, remapped);
                    if pushed.filtering_node_id.is_some() {
                        let mut rebuilt = projection.clone();
                        rebuilt.input = pushed.node_id;
                        let new_id = self.push_node(PlanNode::Projection(rebuilt), old_node.facts.clone());
                        return PushedFilterInput { node_id: new_id, ..pushed };
                    }
                    let input_new = self.rewrite_subtree(old_input);
                    return PushedFilterInput {
                        node_id: input_new,
                        filtered_child_node_id: None,
                        filtering_node_id: None,
                    };
                }
            },
            PlanNode::Join(join) => {
                if let Some((side, remapped)) = remap_single_side_join_output_predicate(&predicate, join) {
                    let child_index = join_side_index(side);
                    let pushed = self.push_filter_into_subtree(join.inputs[child_index], remapped);
                    if pushed.filtering_node_id.is_some() {
                        let mut rebuilt = join.clone();
                        rebuilt.inputs = [self.rewrite_subtree(join.inputs[0]), self.rewrite_subtree(join.inputs[1])];
                        rebuilt.inputs[child_index] = pushed.node_id;
                        let new_id = self.push_node(PlanNode::Join(rebuilt), old_node.facts.clone());
                        return PushedFilterInput { node_id: new_id, ..pushed };
                    }
                    let input_new = self.rewrite_subtree(old_input);
                    return PushedFilterInput {
                        node_id: input_new,
                        filtered_child_node_id: None,
                        filtering_node_id: None,
                    };
                }
            },
            PlanNode::Source(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_)
            | PlanNode::MlAlgorithm(_) => {},
        }

        self.wrap_filter_above_subtree(old_input, predicate)
    }

    /// Wraps the subtree in a filter when its source is dimension-sized Parquet
    /// or already resident. A filter over a fact scan would add a pass over
    /// the large input for rows the residual removes later.
    fn wrap_filter_above_subtree(&mut self, old_input: NodeId, predicate: Expr) -> PushedFilterInput {
        if !residual_filter_pushdown_target_supported(self.original, old_input) {
            let input_new = self.rewrite_subtree(old_input);
            return PushedFilterInput { node_id: input_new, filtered_child_node_id: None, filtering_node_id: None };
        }

        let input_new = self.rewrite_subtree(old_input);
        let input_node =
            self.nodes.get(input_new.index()).expect("rewritten join input node must exist before adding filter");
        let facts = input_node.facts.clone();
        let filter_new = self.push_node(PlanNode::Filter(FilterNode { input: input_new, predicate }), facts);
        PushedFilterInput {
            node_id: filter_new,
            filtered_child_node_id: Some(old_input),
            filtering_node_id: Some(filter_new),
        }
    }

    fn plan_reassociation(
        &mut self,
        join: &JoinNode,
        output_schema: SchemaRef,
    ) -> std::result::Result<ReassociationPlan, JoinFilterReassociationRejectedReason> {
        if !parent_join_supported(join) {
            return Err(JoinFilterReassociationRejectedReason::UnsupportedParentJoinShape);
        }
        let preserved_side = parent_preserved_output_side(join)
            .ok_or(JoinFilterReassociationRejectedReason::NonPreservedSideProjection)?;
        let preserved_input = join.inputs[join_side_index(preserved_side)];
        let filtering_input = join.inputs[join_side_index(opposite_side(preserved_side))];
        let lineage =
            PreservedInputLineage::analyze(self.original, preserved_input, &parent_keys(join, preserved_side))?;

        let crossed_node = self
            .original
            .node(lineage.crossed_join_node_id)
            .ok_or(JoinFilterReassociationRejectedReason::MissingKeyLineage)?;
        let PlanNode::Join(crossed_join) = &crossed_node.kind else {
            return Err(JoinFilterReassociationRejectedReason::UnsupportedCrossedJoinShape);
        };
        if crossed_join.spec.join_type != JoinType::Inner
            || crossed_join.spec.null_equality != JoinNullEquality::NullsDoNotMatch
        {
            return Err(JoinFilterReassociationRejectedReason::UnsupportedCrossedJoinShape);
        }

        let filtered_child_old = crossed_join.inputs[join_side_index(lineage.filtered_side)];
        let filtering_new = self.rewrite_subtree(filtering_input);
        let filtered_child_new = self.rewrite_subtree(filtered_child_old);
        let filter_join_new = self.build_filter_join(filtered_child_new, filtering_new, join, preserved_side, &lineage);

        let mut rebuilt_crossed = crossed_join.clone();
        // Reassociation preserves the parent's row multiplicity, while the
        // rebuilt intermediate carries different columns. The crossed join's
        // old statistics describe a different rowset and must not survive.
        rebuilt_crossed.spec.advisory_output_statistics = join.spec.advisory_output_statistics.and_then(|statistics| {
            let statistics = AdvisoryOutputStatistics { bytes: AdvisoryStatistic::Absent, ..statistics };
            (!statistics.is_absent()).then_some(statistics)
        });
        rebuilt_crossed.inputs =
            [self.rewrite_subtree(crossed_join.inputs[0]), self.rewrite_subtree(crossed_join.inputs[1])];
        rebuilt_crossed.inputs[join_side_index(lineage.filtered_side)] = filter_join_new;
        let crossed_new = self.push_node(PlanNode::Join(rebuilt_crossed), crossed_node.facts.clone());
        let preserved_new = if let Some(projection_node_id) = lineage.projection_node_id {
            let projection_node =
                self.original.node(projection_node_id).expect("lineage projection node must exist after analysis");
            let PlanNode::Projection(projection) = &projection_node.kind else {
                unreachable!("lineage projection id must reference projection");
            };
            let mut projection = projection.clone();
            projection.input = crossed_new;
            self.push_node(PlanNode::Projection(projection), projection_node.facts.clone())
        } else {
            crossed_new
        };

        let parent_projection = parent_projection_for_preserved_side(join, preserved_side, &output_schema)
            .ok_or(JoinFilterReassociationRejectedReason::NonPreservedSideProjection)?;
        let new_id = self.push_node(
            PlanNode::Projection(ProjectionNode::new(preserved_new, parent_projection)),
            PlanFacts::new(output_schema),
        );
        Ok(ReassociationPlan {
            new_id,
            crossed_join_node_id: lineage.crossed_join_node_id,
            filtered_child_node_id: filtered_child_old,
            filtering_node_id: filtering_input,
        })
    }

    fn build_filter_join(
        &mut self,
        filtered_child_new: NodeId,
        filtering_new: NodeId,
        parent_join: &JoinNode,
        preserved_side: JoinSide,
        lineage: &PreservedInputLineage,
    ) -> NodeId {
        let keys = parent_join
            .spec
            .keys
            .iter()
            .zip(&lineage.filtered_child_columns)
            .map(|(parent_key, filtered_column)| {
                let filtering_column = match preserved_side {
                    JoinSide::Left => &parent_key.right,
                    JoinSide::Right => &parent_key.left,
                };
                JoinKey::new(
                    ColumnRef::input(0, filtered_column.name(), filtered_column.display_name()),
                    ColumnRef::input(1, filtering_column.name(), filtering_column.display_name()),
                )
            })
            .collect::<Vec<_>>();
        let filtered_schema = Arc::clone(self.nodes[filtered_child_new.index()].output_schema());
        let projection = filtered_schema
            .fields()
            .iter()
            .map(|field| JoinOutputColumn::left(ColumnRef::input(0, field.name(), field.name()), field.name()))
            .collect::<Vec<_>>();
        let spec = JoinSpec::new(JoinType::Inner, keys)
            .with_output_projection(projection)
            .with_null_equality(JoinNullEquality::NullsDoNotMatch);
        self.push_node(
            PlanNode::Join(JoinNode { inputs: [filtered_child_new, filtering_new], spec }),
            PlanFacts::new(filtered_schema),
        )
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }

    fn consumer_count(&self, node_id: NodeId) -> usize {
        self.consumer_counts[node_id.index()]
    }
}

fn node_consumer_counts(plan: &QueryPlan) -> Vec<usize> {
    let mut consumer_counts = vec![0; plan.nodes().len()];
    for node in plan.nodes() {
        for input in node.input_ids() {
            consumer_counts[input.index()] += 1;
        }
    }
    consumer_counts[plan.sink().input().index()] += 1;
    consumer_counts
}

struct ReassociationPlan {
    new_id: NodeId,
    crossed_join_node_id: NodeId,
    filtered_child_node_id: NodeId,
    filtering_node_id: NodeId,
}

struct ResidualFilterPushdownPlan {
    new_id: NodeId,
    filtered_child_node_id: Option<NodeId>,
    filtering_node_id: Option<NodeId>,
}

struct PushedFilterInput {
    node_id: NodeId,
    filtered_child_node_id: Option<NodeId>,
    filtering_node_id: Option<NodeId>,
}

/// Where the parent join's preserved-side keys come from below the crossed
/// join.
struct PreservedInputLineage {
    /// Renaming projection between the parent and the crossed join, rebuilt
    /// above the new crossed join when present.
    projection_node_id: Option<NodeId>,
    crossed_join_node_id: NodeId,
    /// Child of the crossed join that owns every key; the new filter join is
    /// placed on this side.
    filtered_side: JoinSide,
    /// The parent's keys restated as that child's columns, in key order.
    filtered_child_columns: Vec<ColumnRef>,
}

impl PreservedInputLineage {
    fn analyze(
        plan: &QueryPlan,
        preserved_input: NodeId,
        required_columns: &[ColumnRef],
    ) -> std::result::Result<Self, JoinFilterReassociationRejectedReason> {
        let node = plan.node(preserved_input).ok_or(JoinFilterReassociationRejectedReason::MissingKeyLineage)?;
        match &node.kind {
            PlanNode::Projection(projection) => {
                let columns = required_columns
                    .iter()
                    .map(|column| {
                        let expr = unique_projection_expr_for_output(projection, column.name())
                            .ok_or(JoinFilterReassociationRejectedReason::MissingKeyLineage)?;
                        let projected = simple_projection_column(expr)
                            .ok_or(JoinFilterReassociationRejectedReason::MissingKeyLineage)?;
                        Ok(projected.clone())
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let mut lineage = Self::analyze_join_output(plan, projection.input, &columns)?;
                lineage.projection_node_id = Some(preserved_input);
                Ok(lineage)
            },
            PlanNode::Join(_) => Self::analyze_join_output(plan, preserved_input, required_columns),
            _ => Err(JoinFilterReassociationRejectedReason::UnsupportedPreservedInputShape),
        }
    }

    fn analyze_join_output(
        plan: &QueryPlan,
        crossed_join_node_id: NodeId,
        required_columns: &[ColumnRef],
    ) -> std::result::Result<Self, JoinFilterReassociationRejectedReason> {
        let node = plan.node(crossed_join_node_id).ok_or(JoinFilterReassociationRejectedReason::MissingKeyLineage)?;
        let PlanNode::Join(join) = &node.kind else {
            return Err(JoinFilterReassociationRejectedReason::UnsupportedPreservedInputShape);
        };
        let mut filtered_side = None;
        let mut filtered_child_columns = Vec::with_capacity(required_columns.len());
        for column in required_columns {
            let output = unique_join_output_for_name(join, column.name())
                .ok_or(JoinFilterReassociationRejectedReason::MissingKeyLineage)?;
            match filtered_side {
                Some(side) if side != output.side => {
                    return Err(JoinFilterReassociationRejectedReason::AmbiguousKeyLineage);
                },
                Some(_) => {},
                None => filtered_side = Some(output.side),
            }
            filtered_child_columns.push(output.column.clone());
        }
        Ok(Self {
            projection_node_id: None,
            crossed_join_node_id,
            filtered_side: filtered_side.ok_or(JoinFilterReassociationRejectedReason::MissingKeyLineage)?,
            filtered_child_columns,
        })
    }
}

fn parent_join_supported(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_none()
        && join.spec.null_equality == JoinNullEquality::NullsDoNotMatch
}

fn residual_pushdown_join_supported(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && join.spec.residual_predicate.is_some()
        && join.spec.null_equality == JoinNullEquality::NullsDoNotMatch
}

fn residual_filter_pushdown_target_supported(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(_) => OptimizerSourceTargetPolicy::default()
                .target_is_dimension_sized(estimated_output_rows(plan, node_id), estimated_output_bytes(plan, node_id)),
            SourceNode::GpuResident(_) | SourceNode::LiteralBatch(_) => true,
        },
        PlanNode::Filter(filter) => residual_filter_pushdown_target_supported(plan, filter.input),
        PlanNode::Projection(projection) => residual_filter_pushdown_target_supported(plan, projection.input),
        PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
    }
}

fn input_has_top_filter(plan: &QueryPlan, node_id: NodeId, predicate: &Expr) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    matches!(&node.kind, PlanNode::Filter(filter) if filter.predicate == *predicate)
}

/// A predicate on `side` that every row matching `expr` satisfies, or `None`.
/// A term over `side` alone is necessary as it stands. A comparison across the
/// two sides becomes a range test when the other side's values are bounded
/// (see [`derive_side_scalar_range_filter`]). Conjuncts contribute
/// independently; a disjunction contributes only when both branches do, as the
/// OR of the two, since a matching row satisfies at least one branch.
fn derive_side_necessary_filter(
    plan: &QueryPlan,
    join: &JoinNode,
    expr: &Expr,
    side: JoinSide,
    resolver: &dyn dimension_pruning::LocalParquetPruningResolver,
) -> Option<Expr> {
    if expression_scope(plan, join, expr) == ExpressionScope::Side(side) {
        return Some(strip_input_qualifiers(expr));
    }
    if let Some(filter) = derive_side_scalar_range_filter(plan, join, expr, side, resolver) {
        return Some(filter);
    }

    match expr {
        Expr::Binary { left, op: BinaryOp::And, right, output_type } => {
            let left_filter = derive_side_necessary_filter(plan, join, left, side, resolver);
            let right_filter = derive_side_necessary_filter(plan, join, right, side, resolver);
            combine_optional_filters(left_filter, BinaryOp::And, right_filter, output_type.clone())
        },
        Expr::Binary { left, op: BinaryOp::Or, right, output_type } => {
            let left_filter = derive_side_necessary_filter(plan, join, left, side, resolver)?;
            let right_filter = derive_side_necessary_filter(plan, join, right, side, resolver)?;
            Some(Expr::Binary {
                left: Box::new(left_filter),
                op: BinaryOp::Or,
                right: Box::new(right_filter),
                output_type: output_type.clone(),
            })
        },
        _ => None,
    }
}

/// Range test on `side` implied by a comparison between `side` and the other
/// side, when the other side's expression has a known integer range over its
/// filtered subtree (Parquet footer statistics through `resolver`, bounded by
/// `OptimizerSourceTargetPolicy::residual_scalar_range_max_rows`). `Eq`
/// yields the range envelope; an inequality yields the one bound every match
/// satisfies. The literal takes the target's type, and a range that does not
/// fit that type yields `None`.
fn derive_side_scalar_range_filter(
    plan: &QueryPlan,
    join: &JoinNode,
    expr: &Expr,
    side: JoinSide,
    resolver: &dyn dimension_pruning::LocalParquetPruningResolver,
) -> Option<Expr> {
    let Expr::Binary { left, op, right, .. } = expr else {
        return None;
    };
    if !matches!(op, BinaryOp::Eq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq) {
        return None;
    }

    let left_scope = expression_scope(plan, join, left);
    let right_scope = expression_scope(plan, join, right);
    let opposite = opposite_side(side);
    let (target_expr, scalar_expr, scalar_input, target_on_left) = match (left_scope, right_scope) {
        (ExpressionScope::Side(left_side), ExpressionScope::Side(right_side))
            if left_side == side && right_side == opposite =>
        {
            (left.as_ref(), right.as_ref(), join.inputs[join_side_index(opposite)], true)
        },
        (ExpressionScope::Side(left_side), ExpressionScope::Side(right_side))
            if right_side == side && left_side == opposite =>
        {
            (right.as_ref(), left.as_ref(), join.inputs[join_side_index(opposite)], false)
        },
        _ => return None,
    };

    let target_expr = strip_input_qualifiers(target_expr);
    let target_type = infer_expr_type(plan, join.inputs[join_side_index(side)], &target_expr)?;
    let scalar_expr = strip_input_qualifiers(scalar_expr);
    let scalar_range = dimension_pruning::derive_i64_expr_range_from_filtered_subtree(
        plan,
        scalar_input,
        &scalar_expr,
        OptimizerSourceTargetPolicy::default().residual_scalar_range_max_rows(),
        resolver,
    )??;
    scalar_range_necessary_filter(target_expr, *op, target_on_left, scalar_range, &target_type)
}

fn scalar_range_necessary_filter(
    target_expr: Expr,
    op: BinaryOp,
    target_on_left: bool,
    range: dimension_pruning::KeyRange,
    target_type: &ArrowDataType,
) -> Option<Expr> {
    match (target_on_left, op) {
        (_, BinaryOp::Eq) => range_envelope_filter(target_expr, range, target_type),
        (true, BinaryOp::Gt) => {
            typed_i64_literal(range.min, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::Gt, literal))
        },
        (true, BinaryOp::GtEq) => {
            typed_i64_literal(range.min, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::GtEq, literal))
        },
        (true, BinaryOp::Lt) => {
            typed_i64_literal(range.max, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::Lt, literal))
        },
        (true, BinaryOp::LtEq) => {
            typed_i64_literal(range.max, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::LtEq, literal))
        },
        (false, BinaryOp::Gt) => {
            typed_i64_literal(range.max, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::Lt, literal))
        },
        (false, BinaryOp::GtEq) => {
            typed_i64_literal(range.max, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::LtEq, literal))
        },
        (false, BinaryOp::Lt) => {
            typed_i64_literal(range.min, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::Gt, literal))
        },
        (false, BinaryOp::LtEq) => {
            typed_i64_literal(range.min, target_type).map(|literal| Expr::binary(target_expr, BinaryOp::GtEq, literal))
        },
        (_, _) => None,
    }
}

fn range_envelope_filter(
    target_expr: Expr,
    range: dimension_pruning::KeyRange,
    target_type: &ArrowDataType,
) -> Option<Expr> {
    let min = typed_i64_literal(range.min, target_type)?;
    let max = typed_i64_literal(range.max, target_type)?;
    if range.min == range.max {
        return Some(Expr::binary(target_expr, BinaryOp::Eq, min));
    }
    Some(Expr::binary(
        Expr::binary(target_expr.clone(), BinaryOp::GtEq, min),
        BinaryOp::And,
        Expr::binary(target_expr, BinaryOp::LtEq, max),
    ))
}

fn typed_i64_literal(value: i64, data_type: &ArrowDataType) -> Option<Expr> {
    let literal = match data_type {
        ArrowDataType::Int8 => LiteralValue::Int8(i8::try_from(value).ok()?),
        ArrowDataType::Int16 => LiteralValue::Int16(i16::try_from(value).ok()?),
        ArrowDataType::Int32 => LiteralValue::Int32(i32::try_from(value).ok()?),
        ArrowDataType::Int64 => LiteralValue::Int64(value),
        ArrowDataType::UInt8 => LiteralValue::UInt8(u8::try_from(value).ok()?),
        ArrowDataType::UInt16 => LiteralValue::UInt16(u16::try_from(value).ok()?),
        ArrowDataType::UInt32 => LiteralValue::UInt32(u32::try_from(value).ok()?),
        ArrowDataType::UInt64 => LiteralValue::UInt64(u64::try_from(value).ok()?),
        ArrowDataType::Date32 => LiteralValue::Date32(i32::try_from(value).ok()?),
        _ => return None,
    };
    Some(Expr::literal(literal))
}

fn infer_expr_type(plan: &QueryPlan, input: NodeId, expr: &Expr) -> Option<ArrowDataType> {
    infer_expr_type_with_schema(expr, plan.node(input)?.output_schema())
}

/// Result type of `expr` over `schema`, used to type range literals. `None`
/// for forms whose type depends on execution (`SUBSTRING`, `CONCAT`, `CASE`),
/// which then receive no range filter.
fn infer_expr_type_with_schema(expr: &Expr, schema: &SchemaRef) -> Option<ArrowDataType> {
    match expr {
        Expr::Column(column) => Some(schema.field_with_name(column.name()).ok()?.data_type().clone()),
        Expr::Literal(value) => Some(value.data_type()),
        Expr::Alias { expr, .. } => infer_expr_type_with_schema(expr, schema),
        Expr::Cast { target_type, .. } | Expr::TryCast { target_type, .. } => Some(target_type.clone()),
        Expr::Binary { left, op, right, output_type } => {
            infer_binary_expr_type(left, *op, right, output_type.as_ref(), schema)
        },
        Expr::Not(_)
        | Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::Like { .. }
        | Expr::InList { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. } => Some(ArrowDataType::Boolean),
        Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Trim { expr, .. }
        | Expr::LTrim { expr, .. }
        | Expr::RTrim { expr, .. }
        | Expr::Abs(expr) => infer_expr_type_with_schema(expr, schema),
        Expr::StringLength { output_type, .. }
        | Expr::RegexpReplace { output_type, .. }
        | Expr::UnaryMath { output_type, .. }
        | Expr::Round { output_type, .. }
        | Expr::DateTrunc { output_type, .. } => Some(output_type.clone()),
        Expr::Extract { .. } => Some(ArrowDataType::Int32),
        Expr::Substring { .. } | Expr::Concat(_) | Expr::Case { .. } => None,
    }
}

fn infer_binary_expr_type(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    output_type: Option<&ArrowDataType>,
    schema: &SchemaRef,
) -> Option<ArrowDataType> {
    if let Some(output_type) = output_type {
        return Some(output_type.clone());
    }
    if matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq) {
        return Some(ArrowDataType::Boolean);
    }
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return Some(ArrowDataType::Boolean);
    }
    let left_interval = is_interval_literal(left);
    let right_interval = is_interval_literal(right);
    let left_type = (!left_interval).then(|| infer_expr_type_with_schema(left, schema)).flatten();
    let right_type = (!right_interval).then(|| infer_expr_type_with_schema(right, schema)).flatten();
    match (op, left_interval, right_interval, left_type, right_type) {
        (BinaryOp::Add | BinaryOp::Subtract, false, true, Some(data_type), _) if is_date_or_timestamp(&data_type) => {
            Some(data_type)
        },
        (BinaryOp::Add, true, false, _, Some(data_type)) if is_date_or_timestamp(&data_type) => Some(data_type),
        (_, _, _, Some(data_type), _) => Some(data_type),
        _ => None,
    }
}

fn is_interval_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::IntervalMonthDayNano { .. }))
}

fn is_date_or_timestamp(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Date32 | ArrowDataType::Timestamp(_, None))
}

fn remap_projection_predicate(predicate: &Expr, projection: &ProjectionNode) -> Option<Expr> {
    remap_unqualified_columns(predicate, &mut |column| {
        let expr = unique_projection_expr_for_output(projection, column.name())?;
        simple_projection_column(expr).map(|input| Expr::column(input.name()))
    })
}

fn remap_single_side_join_output_predicate(predicate: &Expr, join: &JoinNode) -> Option<(JoinSide, Expr)> {
    if join.spec.join_type != JoinType::Inner
        || join.spec.null_equality != JoinNullEquality::NullsDoNotMatch
        || join.spec.output_projection.is_empty()
    {
        return None;
    }

    let mut side = None;
    let remapped = remap_unqualified_columns(predicate, &mut |column| {
        let output = unique_join_output_for_name(join, column.name())?;
        match side {
            Some(existing) if existing != output.side => return None,
            Some(_) => {},
            None => side = Some(output.side),
        }
        Some(Expr::column(output.column.name()))
    })?;
    side.map(|side| (side, remapped))
}

// Deliberately broader than the similarly named helper in `dimension_pruning`:
// reassociation rebuilds supported expression trees, while pruning keeps a
// tighter proof boundary for source-pushdown safety.
fn remap_unqualified_columns(expr: &Expr, map_column: &mut impl FnMut(&ColumnRef) -> Option<Expr>) -> Option<Expr> {
    match expr {
        Expr::Column(column) if column.qualifier().is_none() => map_column(column),
        Expr::Column(_) => None,
        Expr::Literal(value) => Some(Expr::Literal(value.clone())),
        Expr::Not(expr) => Some(Expr::logical_not(remap_unqualified_columns(expr, map_column)?)),
        Expr::IsNull(expr) => Some(Expr::is_null(remap_unqualified_columns(expr, map_column)?)),
        Expr::IsNotNull(expr) => Some(Expr::is_not_null(remap_unqualified_columns(expr, map_column)?)),
        Expr::Cast { expr, target_type } => {
            Some(Expr::cast(remap_unqualified_columns(expr, map_column)?, target_type.clone()))
        },
        Expr::TryCast { expr, target_type } => {
            Some(Expr::try_cast(remap_unqualified_columns(expr, map_column)?, target_type.clone()))
        },
        Expr::Binary { left, op, right, output_type } => Some(Expr::Binary {
            left: Box::new(remap_unqualified_columns(left, map_column)?),
            op: *op,
            right: Box::new(remap_unqualified_columns(right, map_column)?),
            output_type: output_type.clone(),
        }),
        Expr::Like { expr, pattern, escape, negated } => Some(Expr::Like {
            expr: Box::new(remap_unqualified_columns(expr, map_column)?),
            pattern: pattern.clone(),
            escape: escape.clone(),
            negated: *negated,
        }),
        Expr::Substring { expr, start, length } => {
            Some(Expr::substring(remap_unqualified_columns(expr, map_column)?, *start, *length))
        },
        Expr::Lower(expr) => Some(Expr::lower(remap_unqualified_columns(expr, map_column)?)),
        Expr::Upper(expr) => Some(Expr::upper(remap_unqualified_columns(expr, map_column)?)),
        Expr::Trim { expr, trim_chars } => {
            Some(Expr::trim(remap_unqualified_columns(expr, map_column)?, trim_chars.clone()))
        },
        Expr::LTrim { expr, trim_chars } => {
            Some(Expr::ltrim(remap_unqualified_columns(expr, map_column)?, trim_chars.clone()))
        },
        Expr::RTrim { expr, trim_chars } => {
            Some(Expr::rtrim(remap_unqualified_columns(expr, map_column)?, trim_chars.clone()))
        },
        Expr::StringContains { expr, pattern } => {
            Some(Expr::string_contains(remap_unqualified_columns(expr, map_column)?, pattern.clone()))
        },
        Expr::StartsWith { expr, prefix } => {
            Some(Expr::starts_with(remap_unqualified_columns(expr, map_column)?, prefix.clone()))
        },
        Expr::EndsWith { expr, suffix } => {
            Some(Expr::ends_with(remap_unqualified_columns(expr, map_column)?, suffix.clone()))
        },
        Expr::StringLength { expr, output_type } => {
            Some(Expr::string_length(remap_unqualified_columns(expr, map_column)?, output_type.clone()))
        },
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => Some(Expr::regexp_replace(
            remap_unqualified_columns(expr, map_column)?,
            pattern.clone(),
            replacement.clone(),
            output_type.clone(),
        )),
        Expr::Concat(exprs) => Some(Expr::concat(
            exprs.iter().map(|expr| remap_unqualified_columns(expr, map_column)).collect::<Option<Vec<_>>>()?,
        )),
        Expr::Abs(expr) => Some(Expr::abs(remap_unqualified_columns(expr, map_column)?)),
        Expr::UnaryMath { op, expr, output_type } => {
            Some(Expr::unary_math(*op, remap_unqualified_columns(expr, map_column)?, output_type.clone()))
        },
        Expr::Round { expr, decimal_places, output_type } => {
            Some(Expr::round(remap_unqualified_columns(expr, map_column)?, *decimal_places, output_type.clone()))
        },
        Expr::InList { expr, list, negated } => Some(Expr::InList {
            expr: Box::new(remap_unqualified_columns(expr, map_column)?),
            list: list.iter().map(|expr| remap_unqualified_columns(expr, map_column)).collect::<Option<Vec<_>>>()?,
            negated: *negated,
        }),
        Expr::IsDistinctFrom { left, right } => Some(Expr::is_distinct_from(
            remap_unqualified_columns(left, map_column)?,
            remap_unqualified_columns(right, map_column)?,
        )),
        Expr::IsNotDistinctFrom { left, right } => Some(Expr::is_not_distinct_from(
            remap_unqualified_columns(left, map_column)?,
            remap_unqualified_columns(right, map_column)?,
        )),
        Expr::Case { branches, else_expr } => Some(Expr::case(
            branches
                .iter()
                .map(|branch| {
                    Some(CaseWhen::new(
                        remap_unqualified_columns(&branch.when, map_column)?,
                        remap_unqualified_columns(&branch.then, map_column)?,
                    ))
                })
                .collect::<Option<Vec<_>>>()?,
            remap_unqualified_columns(else_expr, map_column)?,
        )),
        Expr::Extract { field, expr } => Some(Expr::extract(*field, remap_unqualified_columns(expr, map_column)?)),
        Expr::DateTrunc { field, expr, output_type } => {
            Some(Expr::date_trunc(*field, remap_unqualified_columns(expr, map_column)?, output_type.clone()))
        },
        Expr::Alias { expr, alias } => Some(remap_unqualified_columns(expr, map_column)?.alias(alias.clone())),
    }
}

fn combine_optional_filters(
    left: Option<Expr>,
    op: BinaryOp,
    right: Option<Expr>,
    output_type: Option<ArrowDataType>,
) -> Option<Expr> {
    match (left, right) {
        (Some(left), Some(right)) => {
            Some(Expr::Binary { left: Box::new(left), op, right: Box::new(right), output_type })
        },
        (Some(expr), None) | (None, Some(expr)) => Some(expr),
        (None, None) => None,
    }
}

/// Which join input an expression's columns come from. A qualified column
/// resolves by input index; an unqualified name resolves only when a single
/// input carries it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpressionScope {
    NoColumns,
    Side(JoinSide),
    MixedOrUnknown,
}

fn expression_scope(plan: &QueryPlan, join: &JoinNode, expr: &Expr) -> ExpressionScope {
    match expr {
        Expr::Column(column) => column_scope(plan, join, column),
        Expr::Literal(_) => ExpressionScope::NoColumns,
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Trim { expr, .. }
        | Expr::LTrim { expr, .. }
        | Expr::RTrim { expr, .. }
        | Expr::StringContains { expr, .. }
        | Expr::StartsWith { expr, .. }
        | Expr::EndsWith { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Abs(expr)
        | Expr::UnaryMath { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. } => expression_scope(plan, join, expr),
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            merge_expression_scopes(expression_scope(plan, join, left), expression_scope(plan, join, right))
        },
        Expr::InList { expr, list, .. } => list.iter().fold(expression_scope(plan, join, expr), |scope, item| {
            merge_expression_scopes(scope, expression_scope(plan, join, item))
        }),
        Expr::Concat(exprs) => exprs.iter().fold(ExpressionScope::NoColumns, |scope, expr| {
            merge_expression_scopes(scope, expression_scope(plan, join, expr))
        }),
        Expr::Case { branches, else_expr } => {
            branches.iter().fold(expression_scope(plan, join, else_expr), |scope, branch| {
                merge_expression_scopes(
                    scope,
                    merge_expression_scopes(
                        expression_scope(plan, join, &branch.when),
                        expression_scope(plan, join, &branch.then),
                    ),
                )
            })
        },
    }
}

fn column_scope(plan: &QueryPlan, join: &JoinNode, column: &ColumnRef) -> ExpressionScope {
    match column.qualifier() {
        Some(ColumnQualifier::Input { index: 0 }) => ExpressionScope::Side(JoinSide::Left),
        Some(ColumnQualifier::Input { index: 1 }) => ExpressionScope::Side(JoinSide::Right),
        _ => unqualified_column_scope(plan, join, column.name()),
    }
}

fn unqualified_column_scope(plan: &QueryPlan, join: &JoinNode, column_name: &str) -> ExpressionScope {
    let left = plan.node(join.inputs[0]).is_some_and(|node| node.output_schema().field_with_name(column_name).is_ok());
    let right = plan.node(join.inputs[1]).is_some_and(|node| node.output_schema().field_with_name(column_name).is_ok());
    match (left, right) {
        (true, false) => ExpressionScope::Side(JoinSide::Left),
        (false, true) => ExpressionScope::Side(JoinSide::Right),
        (false, false) | (true, true) => ExpressionScope::MixedOrUnknown,
    }
}

fn merge_expression_scopes(left: ExpressionScope, right: ExpressionScope) -> ExpressionScope {
    match (left, right) {
        (ExpressionScope::MixedOrUnknown, _) | (_, ExpressionScope::MixedOrUnknown) => ExpressionScope::MixedOrUnknown,
        (ExpressionScope::NoColumns, other) | (other, ExpressionScope::NoColumns) => other,
        (ExpressionScope::Side(left), ExpressionScope::Side(right)) if left == right => ExpressionScope::Side(left),
        (ExpressionScope::Side(_), ExpressionScope::Side(_)) => ExpressionScope::MixedOrUnknown,
    }
}

fn strip_input_qualifiers(expr: &Expr) -> Expr {
    match expr {
        Expr::Column(column) => Expr::column(column.name()),
        Expr::Literal(value) => Expr::Literal(value.clone()),
        Expr::Not(expr) => Expr::logical_not(strip_input_qualifiers(expr)),
        Expr::IsNull(expr) => Expr::is_null(strip_input_qualifiers(expr)),
        Expr::IsNotNull(expr) => Expr::is_not_null(strip_input_qualifiers(expr)),
        Expr::Cast { expr, target_type } => Expr::cast(strip_input_qualifiers(expr), target_type.clone()),
        Expr::TryCast { expr, target_type } => Expr::try_cast(strip_input_qualifiers(expr), target_type.clone()),
        Expr::Binary { left, op, right, output_type } => Expr::Binary {
            left: Box::new(strip_input_qualifiers(left)),
            op: *op,
            right: Box::new(strip_input_qualifiers(right)),
            output_type: output_type.clone(),
        },
        Expr::Like { expr, pattern, escape, negated } => Expr::Like {
            expr: Box::new(strip_input_qualifiers(expr)),
            pattern: pattern.clone(),
            escape: escape.clone(),
            negated: *negated,
        },
        Expr::Substring { expr, start, length } => Expr::substring(strip_input_qualifiers(expr), *start, *length),
        Expr::Lower(expr) => Expr::lower(strip_input_qualifiers(expr)),
        Expr::Upper(expr) => Expr::upper(strip_input_qualifiers(expr)),
        Expr::Trim { expr, trim_chars } => Expr::trim(strip_input_qualifiers(expr), trim_chars.clone()),
        Expr::LTrim { expr, trim_chars } => Expr::ltrim(strip_input_qualifiers(expr), trim_chars.clone()),
        Expr::RTrim { expr, trim_chars } => Expr::rtrim(strip_input_qualifiers(expr), trim_chars.clone()),
        Expr::StringContains { expr, pattern } => Expr::string_contains(strip_input_qualifiers(expr), pattern.clone()),
        Expr::StartsWith { expr, prefix } => Expr::starts_with(strip_input_qualifiers(expr), prefix.clone()),
        Expr::EndsWith { expr, suffix } => Expr::ends_with(strip_input_qualifiers(expr), suffix.clone()),
        Expr::StringLength { expr, output_type } => {
            Expr::string_length(strip_input_qualifiers(expr), output_type.clone())
        },
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => Expr::regexp_replace(
            strip_input_qualifiers(expr),
            pattern.clone(),
            replacement.clone(),
            output_type.clone(),
        ),
        Expr::Concat(exprs) => Expr::concat(exprs.iter().map(strip_input_qualifiers).collect()),
        Expr::Abs(expr) => Expr::abs(strip_input_qualifiers(expr)),
        Expr::UnaryMath { op, expr, output_type } => {
            Expr::unary_math(*op, strip_input_qualifiers(expr), output_type.clone())
        },
        Expr::Round { expr, decimal_places, output_type } => {
            Expr::round(strip_input_qualifiers(expr), *decimal_places, output_type.clone())
        },
        Expr::InList { expr, list, negated } => Expr::InList {
            expr: Box::new(strip_input_qualifiers(expr)),
            list: list.iter().map(strip_input_qualifiers).collect(),
            negated: *negated,
        },
        Expr::IsDistinctFrom { left, right } => {
            Expr::is_distinct_from(strip_input_qualifiers(left), strip_input_qualifiers(right))
        },
        Expr::IsNotDistinctFrom { left, right } => {
            Expr::is_not_distinct_from(strip_input_qualifiers(left), strip_input_qualifiers(right))
        },
        Expr::Case { branches, else_expr } => Expr::case(
            branches
                .iter()
                .map(|branch| CaseWhen {
                    when: strip_input_qualifiers(&branch.when),
                    then: strip_input_qualifiers(&branch.then),
                })
                .collect(),
            strip_input_qualifiers(else_expr),
        ),
        Expr::Extract { field, expr } => Expr::extract(*field, strip_input_qualifiers(expr)),
        Expr::DateTrunc { field, expr, output_type } => {
            Expr::date_trunc(*field, strip_input_qualifiers(expr), output_type.clone())
        },
        Expr::Alias { expr, alias } => strip_input_qualifiers(expr).alias(alias.clone()),
    }
}

/// The one side every output of `join` comes from, or `None` when outputs mix
/// sides or the projection is empty.
fn parent_preserved_output_side(join: &JoinNode) -> Option<JoinSide> {
    let mut side = None;
    for output in &join.spec.output_projection {
        match side {
            Some(existing) if existing != output.side => return None,
            Some(_) => {},
            None => side = Some(output.side),
        }
    }
    side
}

/// `join` with its output projection narrowed to what `projection` keeps, so
/// a join whose own output mixes sides still qualifies when the projection
/// above it reads one side.
fn projected_preserved_side_join(join: &JoinNode, projection: &ProjectionNode) -> Option<JoinNode> {
    let mut output_projection = Vec::with_capacity(projection.outputs.len());
    let mut side = None;
    for output in &projection.outputs {
        let column = simple_projection_column(&output.expr)?;
        let join_output = unique_join_output_for_name(join, column.name())?;
        match side {
            Some(existing) if existing != join_output.side => return None,
            Some(_) => {},
            None => side = Some(join_output.side),
        }
        output_projection.push(JoinOutputColumn {
            side: join_output.side,
            column: join_output.column.clone(),
            output_name: output.field.name().clone(),
        });
    }
    side?;
    let mut projected_join = join.clone();
    projected_join.spec.output_projection = output_projection;
    Some(projected_join)
}

fn parent_keys(join: &JoinNode, side: JoinSide) -> Vec<ColumnRef> {
    join.spec
        .keys
        .iter()
        .map(|key| match side {
            JoinSide::Left => key.left.clone(),
            JoinSide::Right => key.right.clone(),
        })
        .collect()
}

fn parent_projection_for_preserved_side(
    join: &JoinNode,
    side: JoinSide,
    output_schema: &SchemaRef,
) -> Option<Vec<ProjectionOutput>> {
    if join.spec.output_projection.len() != output_schema.fields().len() {
        return None;
    }
    join.spec
        .output_projection
        .iter()
        .zip(output_schema.fields().iter())
        .map(|(output, field)| {
            (output.side == side).then(|| {
                ProjectionOutput::new(
                    Expr::column(output.column.name()).alias(output.output_name.clone()),
                    field.as_ref().clone(),
                )
            })
        })
        .collect()
}

fn unique_projection_expr_for_output<'a>(projection: &'a ProjectionNode, output_name: &str) -> Option<&'a Expr> {
    let mut matches = projection.outputs.iter().filter(|output| output.field.name() == output_name);
    let first = matches.next()?;
    matches.next().is_none().then_some(&first.expr)
}

const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

const fn opposite_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::LiteralValue,
        plan::{
            AdvisoryOutputBinding, AdvisoryOutputStatistics, AdvisoryStatistic, JoinKey, JoinSpec, QueryPlanBuilder,
        },
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn residual_pushdown_extracts_side_filters_from_or_of_and_terms() {
        let plan = residual_join_plan(or(
            and(left_eq("l_flag", 1), right_gt("r_price", 10)),
            and(left_eq("l_flag", 2), right_lt("r_price", 5)),
        ));

        let result = optimize_join_filter_reassociation(
            plan,
            &residual_pushdown_options(),
            &dimension_pruning::NoDimensionPruningFacts,
        )
        .unwrap();

        assert!(result.change.changed());
        assert_eq!(result.diagnostics.applied_count, 1);
        let join = root_join(&result.plan);
        let left_filter = filter_node(&result.plan, join.inputs[0]);
        let right_filter = filter_node(&result.plan, join.inputs[1]);
        assert_eq!(left_filter.predicate, or(unqualified_eq("l_flag", 1), unqualified_eq("l_flag", 2)));
        assert_eq!(right_filter.predicate, or(unqualified_gt("r_price", 10), unqualified_lt("r_price", 5)));
        assert!(join.spec.residual_predicate.is_some(), "residual stays in place for correctness");
    }

    #[test]
    fn residual_pushdown_requires_every_or_branch_to_imply_the_side_filter() {
        let residual = or(and(left_eq("l_flag", 1), right_gt("r_price", 10)), right_lt("r_price", 5));
        let plan = residual_join_plan(residual);

        let result = optimize_join_filter_reassociation(
            plan,
            &residual_pushdown_options(),
            &dimension_pruning::NoDimensionPruningFacts,
        )
        .unwrap();

        assert!(result.change.changed());
        let join = root_join(&result.plan);
        assert!(
            !matches!(result.plan.node(join.inputs[0]).map(|node| &node.kind), Some(PlanNode::Filter(_))),
            "left filter is unsafe because one OR branch has no left-side predicate"
        );
        let right_filter = filter_node(&result.plan, join.inputs[1]);
        assert_eq!(right_filter.predicate, or(unqualified_gt("r_price", 10), unqualified_lt("r_price", 5)));
    }

    #[test]
    fn residual_pushdown_is_not_reapplied_when_filter_is_already_present() {
        let plan = residual_join_plan(and(left_eq("l_flag", 1), right_gt("r_price", 10)));

        let first = optimize_join_filter_reassociation(
            plan,
            &residual_pushdown_options(),
            &dimension_pruning::NoDimensionPruningFacts,
        )
        .unwrap();
        assert!(first.change.changed());
        let second = optimize_join_filter_reassociation(
            first.plan,
            &residual_pushdown_options(),
            &dimension_pruning::NoDimensionPruningFacts,
        )
        .unwrap();

        assert!(!second.change.changed());
        assert_eq!(second.diagnostics.applied_count, 0);
    }

    /// The rule rebuilds the plan in its own visit order, so the right source
    /// moves to a new id; its advisory binding moves with it unchanged.
    #[test]
    fn residual_pushdown_keeps_advisory_on_the_renumbered_unchanged_source() {
        let (mut builder, join, right) = residual_join_builder(and(left_eq("l_flag", 1), right_gt("r_price", 10)));
        let statistics =
            AdvisoryOutputStatistics { rows: AdvisoryStatistic::Inexact(100), bytes: AdvisoryStatistic::Inexact(800) };
        builder
            .set_advisory_output(AdvisoryOutputBinding {
                node: right,
                schema: schema(vec![field("r_key"), field("r_price")]),
                frontend: "test frontend".to_owned(),
                frontend_version: "1".to_owned(),
                statistics,
                unknown_reason: None,
            })
            .unwrap();

        let result = optimize_join_filter_reassociation(
            builder.finish_gpu_resident(join).unwrap(),
            &residual_pushdown_options(),
            &dimension_pruning::NoDimensionPruningFacts,
        )
        .unwrap();

        assert!(result.change.changed());
        let [binding] = result.plan.advisory_outputs() else {
            panic!("one advisory binding: {:?}", result.plan.advisory_outputs());
        };
        let node = &result.plan.node(binding.node).unwrap().kind;
        let PlanNode::Source(source) = node else { panic!("binding must stay on a source: {node:?}") };
        assert!(matches!(source.as_ref(), SourceNode::GpuResident(source) if source.input_name == "right"));
        assert_eq!(binding.statistics, statistics);
        assert_eq!(binding.unknown_reason, None);
    }

    fn residual_join_plan(residual: Expr) -> QueryPlan {
        let (builder, join, _right) = residual_join_builder(residual);
        builder.finish_gpu_resident(join).unwrap()
    }

    fn residual_join_builder(residual: Expr) -> (QueryPlanBuilder, NodeId, NodeId) {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(source("left", vec![field("l_key"), field("l_flag")]));
        let right = builder.add_source(source("right", vec![field("r_key"), field("r_price")]));
        let output_schema = schema(vec![field("l_key"), field("l_flag"), field("r_price")]);
        let join = builder
            .add_join(
                left,
                right,
                JoinSpec::new(
                    JoinType::Inner,
                    vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
                )
                .with_residual_predicate(residual)
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
                    JoinOutputColumn::left(ColumnRef::input(0, "l_flag", "l_flag"), "l_flag"),
                    JoinOutputColumn::right(ColumnRef::input(1, "r_price", "r_price"), "r_price"),
                ]),
                output_schema,
            )
            .unwrap();
        (builder, join, right)
    }

    fn residual_pushdown_options() -> JoinFilterReassociationOptions {
        JoinFilterReassociationOptions::default().with_residual_filter_pushdown(true)
    }

    fn root_join(plan: &QueryPlan) -> &JoinNode {
        let PlanNode::Join(join) = &plan.root_node().kind else {
            panic!("root must be a join");
        };
        join
    }

    fn filter_node(plan: &QueryPlan, node_id: NodeId) -> &FilterNode {
        let Some(node) = plan.node(node_id) else {
            panic!("missing node {node_id}");
        };
        let PlanNode::Filter(filter) = &node.kind else {
            panic!("node {node_id} must be a filter");
        };
        filter
    }

    fn source(name: &str, fields: Vec<Field>) -> SourceNode {
        SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
    }

    fn schema(fields: Vec<Field>) -> Arc<Schema> {
        Arc::new(Schema::new(fields))
    }

    fn field(name: &str) -> Field {
        Field::new(name, DataType::Int32, false)
    }

    fn left_eq(name: &str, value: i32) -> Expr {
        Expr::binary(Expr::input_column(0, name, name), BinaryOp::Eq, int(value))
    }

    fn right_gt(name: &str, value: i32) -> Expr {
        Expr::binary(Expr::input_column(1, name, name), BinaryOp::Gt, int(value))
    }

    fn right_lt(name: &str, value: i32) -> Expr {
        Expr::binary(Expr::input_column(1, name, name), BinaryOp::Lt, int(value))
    }

    fn unqualified_eq(name: &str, value: i32) -> Expr {
        Expr::binary(Expr::column(name), BinaryOp::Eq, int(value))
    }

    fn unqualified_gt(name: &str, value: i32) -> Expr {
        Expr::binary(Expr::column(name), BinaryOp::Gt, int(value))
    }

    fn unqualified_lt(name: &str, value: i32) -> Expr {
        Expr::binary(Expr::column(name), BinaryOp::Lt, int(value))
    }

    fn and(left: Expr, right: Expr) -> Expr {
        Expr::binary(left, BinaryOp::And, right)
    }

    fn or(left: Expr, right: Expr) -> Expr {
        Expr::binary(left, BinaryOp::Or, right)
    }

    fn int(value: i32) -> Expr {
        Expr::literal(LiteralValue::Int32(value))
    }
}
