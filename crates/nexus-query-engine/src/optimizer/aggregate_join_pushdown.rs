use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use super::{
    aggregate_join_pushdown_policy::AggregateJoinPushdownPolicy, rewrite, simple_projection_column,
    unique_join_output_for_name,
};
use crate::{
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, LiteralValue},
    plan::{
        AggregateNode, JoinKey, JoinNode, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanFacts, PlanNode,
        ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, SchemaBehavior,
    },
};

/// Stable rule name used in optimizer diagnostics.
pub const AGGREGATE_JOIN_PUSHDOWN_RULE_NAME: &str = "aggregate_join_pushdown";

/// Configuration for reducing aggregates before joins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateJoinPushdownOptions {
    /// Whether the pass is allowed to rewrite eligible aggregate-over-join shapes.
    pub enabled: bool,
}

impl AggregateJoinPushdownOptions {
    /// Returns options with the pass disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for AggregateJoinPushdownOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for aggregate-join pushdown candidates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AggregateJoinPushdownDiagnostics {
    /// Number of candidates considered.
    pub considered_count: usize,
    /// Number of candidates rewritten.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<AggregateJoinPushdownCandidateDiagnostics>,
}

impl AggregateJoinPushdownDiagnostics {
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

/// Per-candidate evidence for aggregate-join pushdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateJoinPushdownCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Aggregate node considered for pushdown.
    pub aggregate_node_id: NodeId,
    /// Join node directly below the aggregate.
    pub join_node_id: Option<NodeId>,
    /// True when the returned plan contains the rewrite.
    pub applied: bool,
    /// Rejection reason when no rewrite was applied.
    pub rejected_reason: Option<AggregateJoinPushdownRejectedReason>,
}

/// Reasons an aggregate-join pushdown candidate was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateJoinPushdownRejectedReason {
    /// The aggregate input is not a join.
    InputNotJoin,
    /// The join type is not currently supported by this pass.
    UnsupportedJoinType,
    /// The join has a residual predicate.
    JoinResidualUnsupported,
    /// Retired: the aggregate is not a grouping-set/rollup aggregate.
    GroupingSetsRequired,
    /// The aggregate contains an unsupported function, DISTINCT, filtered, or non-column expression.
    UnsupportedAggregateExpression,
    /// The aggregate grouping keys are not plain columns.
    UnsupportedGroupExpression,
    /// Aggregate input columns could not be mapped through the join output.
    MissingJoinOutputLineage,
    /// The eligible columns span sides in a pattern not covered by the rewrite.
    UnsupportedSidePattern,
    /// The pushed aggregate would require a missing input field.
    MissingInputField,
    /// Two required pre-aggregate columns would produce the same output name.
    ConflictingPreAggregateOutputName,
}

pub(crate) struct AggregateJoinPushdownResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: AggregateJoinPushdownDiagnostics,
}

pub(crate) fn optimize_aggregate_join_pushdown(
    plan: QueryPlan,
    options: &AggregateJoinPushdownOptions,
) -> AggregateJoinPushdownResult {
    if !options.enabled {
        return AggregateJoinPushdownResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: AggregateJoinPushdownDiagnostics::default(),
        };
    }

    let mut rewriter = AggregateJoinPushdownRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let AggregateJoinPushdownRewriter { diagnostics, .. } = rewriter;
        return AggregateJoinPushdownResult { plan, change: super::NativeOptimizerChange::Unchanged, diagnostics };
    }

    AggregateJoinPushdownResult {
        plan: rewrite::finish_compacted_rewritten_plan(rewriter.nodes, new_root, plan.sink())
            .with_keyed_hash_join_strategy_options_from(&plan),
        change: super::NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    }
}

struct AggregateJoinPushdownRewriter<'a> {
    original: &'a QueryPlan,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    diagnostics: AggregateJoinPushdownDiagnostics,
    changed: bool,
}

impl<'a> AggregateJoinPushdownRewriter<'a> {
    fn new(original: &'a QueryPlan) -> Self {
        Self {
            original,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            diagnostics: AggregateJoinPushdownDiagnostics::default(),
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_id = match &old_node.kind {
            PlanNode::Aggregate(aggregate) => self.rewrite_aggregate(old_id, aggregate, old_node.facts.clone()),
            _ => {
                let new_kind = self.clone_kind_with_rewritten_inputs(&old_node.kind);
                self.push_node(new_kind, old_node.facts.clone())
            },
        };
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn rewrite_aggregate(&mut self, old_id: NodeId, aggregate: &AggregateNode, facts: PlanFacts) -> NodeId {
        let rewritten_input = self.rewrite_subtree(aggregate.input);
        let rewritten_aggregate = AggregateNode { input: rewritten_input, spec: aggregate.spec.clone() };

        let candidate = self.try_rewrite_aggregate_join(old_id, &rewritten_aggregate, &facts);
        match candidate {
            Ok(Some(new_id)) => {
                self.changed = true;
                self.diagnostics.considered_count += 1;
                self.diagnostics.applied_count += 1;
                self.diagnostics.candidates.push(AggregateJoinPushdownCandidateDiagnostics {
                    rule_name: AGGREGATE_JOIN_PUSHDOWN_RULE_NAME,
                    aggregate_node_id: old_id,
                    join_node_id: Some(rewritten_input),
                    applied: true,
                    rejected_reason: None,
                });
                new_id
            },
            Ok(None) => self.push_node(PlanNode::Aggregate(rewritten_aggregate), facts),
            Err(reason) => {
                self.diagnostics.considered_count += 1;
                self.diagnostics.candidates.push(AggregateJoinPushdownCandidateDiagnostics {
                    rule_name: AGGREGATE_JOIN_PUSHDOWN_RULE_NAME,
                    aggregate_node_id: old_id,
                    join_node_id: (reason != AggregateJoinPushdownRejectedReason::InputNotJoin)
                        .then_some(rewritten_input),
                    applied: false,
                    rejected_reason: Some(reason),
                });
                self.push_node(PlanNode::Aggregate(rewritten_aggregate), facts)
            },
        }
    }

    fn try_rewrite_aggregate_join(
        &mut self,
        _aggregate_id: NodeId,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
    ) -> Result<Option<NodeId>, AggregateJoinPushdownRejectedReason> {
        let (join, projection_chain) = self.resolve_aggregate_join_input(aggregate.input)?;
        if join.spec.residual_predicate.is_some() {
            return Err(AggregateJoinPushdownRejectedReason::JoinResidualUnsupported);
        }

        match join.spec.join_type {
            JoinType::Inner => self.rewrite_inner_join_aggregate(aggregate, aggregate_facts, &join, &projection_chain),
            JoinType::Left => {
                self.rewrite_left_join_count_aggregate(aggregate, aggregate_facts, &join, &projection_chain)
            },
            JoinType::LeftSemi => {
                self.rewrite_left_semi_join_aggregate(aggregate, aggregate_facts, &join, &projection_chain)
            },
            _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedJoinType),
        }
        .map(Some)
    }

    /// Walks down from the aggregate's direct input through any [`PlanNode::Projection`]
    /// chain until reaching a [`PlanNode::Join`]. Returns the join and the chain
    /// of intervening projections with the outermost first (closest to the
    /// aggregate) and the innermost last (closest to the join).
    fn resolve_aggregate_join_input(
        &self,
        aggregate_input: NodeId,
    ) -> Result<(JoinNode, Vec<ProjectionNode>), AggregateJoinPushdownRejectedReason> {
        let mut chain = Vec::new();
        let mut current = aggregate_input;
        loop {
            let node = self
                .nodes
                .get(current.index())
                .filter(|node| node.id == current)
                .ok_or(AggregateJoinPushdownRejectedReason::InputNotJoin)?;
            match &node.kind {
                PlanNode::Join(join) => return Ok((join.clone(), chain)),
                PlanNode::Projection(projection) => {
                    let projection = projection.clone();
                    current = projection.input;
                    chain.push(projection);
                },
                _ => return Err(AggregateJoinPushdownRejectedReason::InputNotJoin),
            }
        }
    }

    fn rewrite_inner_join_aggregate(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let lineage = AggregateJoinLineage::new(aggregate, join, projection_chain)?;
        let measure_side = lineage.single_measure_side()?;
        // For inner joins we allow group keys on either side or split across
        // both: the pre-aggregate only carries measure-side group keys, and the
        // final aggregate still sees the other-side keys through the rebuilt
        // join's output projection.
        let preagg_group_columns = lineage.group_columns_for_side(measure_side);
        self.rewrite_join_aggregate_with_preagg(
            aggregate,
            aggregate_facts,
            join,
            measure_side,
            preagg_group_columns,
            &lineage,
            JoinType::Inner,
            false,
        )
    }

    fn rewrite_left_join_count_aggregate(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let lineage = AggregateJoinLineage::new(aggregate, join, projection_chain)?;
        if lineage.single_group_side()? != JoinSide::Left || lineage.single_measure_side()? != JoinSide::Right {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
        }
        if !lineage.measure_columns.iter().all(|measure| {
            measure.side_input_aggregate.function == AggregateFunction::Count
                && matches!(measure.side_input_aggregate.expr, Expr::Column(_))
                && !measure.side_input_aggregate.distinct
                && measure.side_input_aggregate.filter.is_none()
        }) {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
        }

        self.rewrite_join_aggregate_with_preagg(
            aggregate,
            aggregate_facts,
            join,
            JoinSide::Right,
            lineage.group_columns_for_side(JoinSide::Right),
            &lineage,
            JoinType::Left,
            true,
        )
    }

    fn rewrite_left_semi_join_aggregate(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let lineage = AggregateJoinLineage::new(aggregate, join, projection_chain)?;
        // LeftSemi only emits left-side columns, so both measures and groups
        // must live on the left input.
        if lineage.single_measure_side()? != JoinSide::Left || lineage.single_group_side()? != JoinSide::Left {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
        }
        self.rewrite_join_aggregate_with_preagg(
            aggregate,
            aggregate_facts,
            join,
            JoinSide::Left,
            lineage.group_columns_for_side(JoinSide::Left),
            &lineage,
            JoinType::LeftSemi,
            false,
        )
    }

    fn rewrite_join_aggregate_with_preagg(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        preagg_side: JoinSide,
        mut preagg_group_columns: Vec<PreAggregateColumn>,
        lineage: &AggregateJoinLineage,
        join_type: JoinType,
        coalesce_preagg_nulls_to_zero: bool,
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let preagg_input = join.inputs[join_side_index(preagg_side)];
        let preagg_input_schema = self
            .nodes
            .get(preagg_input.index())
            .filter(|node| node.id == preagg_input)
            .map(|node| node.output_schema().clone())
            .ok_or(AggregateJoinPushdownRejectedReason::MissingInputField)?;
        for key in &join.spec.keys {
            let key_column = join_key_for_side(key, preagg_side);
            push_unique_preagg_group_column(
                &mut preagg_group_columns,
                key_column.name(),
                key_column.display_name(),
                &preagg_input_schema,
            )?;
        }

        let preagg_state_aliases =
            preaggregate_state_aliases(aggregate, aggregate_facts.output_schema.as_ref(), &preagg_group_columns);
        let reserved_preagg_aliases =
            preagg_state_aliases.iter().flat_map(|aliases| aliases.iter().cloned()).collect::<Vec<_>>();
        // The aggregate executor requires aggregate measure arguments to be
        // plain column references. When every measure already resolves to a
        // plain column from the measure-side input (the q70/q86 case with no
        // projection chain), skip the materializing projection and have the
        // pre-aggregate read those columns directly. When at least one measure
        // came from a computed projection-chain expression (the q67 case),
        // materialize all measures once under fresh aliases so the pre-aggregate
        // can still reference them as plain columns.
        let needs_materializing_projection =
            lineage.measure_columns.iter().any(|measure| !matches!(measure.side_input_aggregate.expr, Expr::Column(_)));

        let (preagg_input_id, preagg_input_schema_for_spec, measure_input_columns) = if needs_materializing_projection {
            let aggregate_input_schema = self
                .nodes
                .get(aggregate.input.index())
                .filter(|node| node.id == aggregate.input)
                .map(|node| node.output_schema().clone())
                .ok_or(AggregateJoinPushdownRejectedReason::MissingInputField)?;
            let measure_input_aliases = aggregate
                .spec
                .aggr_exprs
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    unique_measure_input_alias(
                        index,
                        aggregate_facts.output_schema.as_ref(),
                        &preagg_group_columns,
                        &reserved_preagg_aliases,
                    )
                })
                .collect::<Vec<_>>();
            let projection_outputs = measure_input_projection_outputs(
                &preagg_input_schema,
                &aggregate_input_schema,
                aggregate,
                &preagg_group_columns,
                lineage,
                &measure_input_aliases,
            )?;
            let projection_node = ProjectionNode::new(preagg_input, projection_outputs);
            let projection_schema = Arc::clone(projection_node.output_schema());
            let projection_id = self.push_node(
                PlanNode::Projection(projection_node),
                PlanFacts::device(Arc::clone(&projection_schema), SchemaBehavior::Projection),
            );
            (projection_id, projection_schema, measure_input_aliases)
        } else {
            // Plain-column measures: pre-aggregate reads each measure column
            // by the side-input name the lineage already resolved.
            let measure_input_columns = lineage
                .measure_columns
                .iter()
                .map(|measure| {
                    let Expr::Column(column) = &measure.side_input_aggregate.expr else {
                        unreachable!("plain-column branch only entered when all measures are plain columns")
                    };
                    column.name().to_string()
                })
                .collect::<Vec<_>>();
            (preagg_input, preagg_input_schema, measure_input_columns)
        };

        let preagg_measure_states = preaggregate_measure_states(
            &preagg_input_schema_for_spec,
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_state_aliases,
            &measure_input_columns,
        )?;
        let preagg_schema = preaggregate_schema(
            &preagg_input_schema_for_spec,
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_group_columns,
            &preagg_measure_states,
        )?;
        let preagg_spec =
            preaggregate_spec(aggregate, &preagg_group_columns, &preagg_measure_states, &measure_input_columns)?;
        let preagg_id = self
            .try_rewrite_preaggregate_through_join_union(preagg_input_id, &preagg_spec, &preagg_schema)
            .unwrap_or_else(|| {
                self.push_node(
                    PlanNode::Aggregate(AggregateNode { input: preagg_input_id, spec: preagg_spec }),
                    PlanFacts::device(Arc::clone(&preagg_schema), SchemaBehavior::Aggregate),
                )
            });

        let new_join =
            self.rewritten_join_after_preagg(lineage, join, preagg_side, preagg_id, &preagg_measure_states, join_type);
        let join_schema = new_join_output_schema(
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_schema,
            &preagg_measure_states,
            preagg_side,
            join_type,
        )?;
        let join_id = self.push_node(PlanNode::Join(new_join), PlanFacts::device(join_schema, SchemaBehavior::Join));
        let final_input = if coalesce_preagg_nulls_to_zero {
            let projection_outputs = coalesce_preagg_projection_outputs(
                aggregate,
                aggregate_facts.output_schema.as_ref(),
                &preagg_measure_states,
            )?;
            let projection_node = ProjectionNode::new(join_id, projection_outputs);
            let projection_schema = Arc::clone(projection_node.output_schema());
            self.push_node(
                PlanNode::Projection(projection_node),
                PlanFacts::device(projection_schema, SchemaBehavior::Projection),
            )
        } else {
            join_id
        };
        let final_rewrite = final_aggregate_rewrite_after_preagg(
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_measure_states,
        )?;
        let final_aggregate_id = self.push_node(
            PlanNode::Aggregate(AggregateNode { input: final_input, spec: final_rewrite.spec }),
            PlanFacts::device(Arc::clone(&final_rewrite.output_schema), SchemaBehavior::Aggregate),
        );
        if let Some(projection_outputs) = final_rewrite.projection_outputs {
            let projection_node = ProjectionNode::new(final_aggregate_id, projection_outputs);
            let projection_schema = Arc::clone(projection_node.output_schema());
            return Ok(self.push_node(
                PlanNode::Projection(projection_node),
                PlanFacts::device(projection_schema, SchemaBehavior::Projection),
            ));
        }
        Ok(final_aggregate_id)
    }

    fn rewritten_join_after_preagg(
        &self,
        lineage: &AggregateJoinLineage,
        join: &JoinNode,
        preagg_side: JoinSide,
        preagg_id: NodeId,
        preagg_measure_states: &[PreAggregateMeasureState],
        join_type: JoinType,
    ) -> JoinNode {
        let other_side = opposite_side(preagg_side);
        let inputs = match preagg_side {
            JoinSide::Left => [preagg_id, join.inputs[join_side_index(other_side)]],
            JoinSide::Right => [join.inputs[join_side_index(other_side)], preagg_id],
        };
        let keys =
            join.spec.keys.iter().map(|key| rewritten_join_key_after_preagg(key, preagg_side)).collect::<Vec<_>>();
        let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
        let mut output_projection = Vec::with_capacity(lineage.group_columns.len() + state_count);
        for group in &lineage.group_columns {
            let side = group.side;
            let join_input_index =
                if side == preagg_side { join_side_index(preagg_side) } else { join_side_index(other_side) };
            let source_name = if side == preagg_side { group.output_name.as_str() } else { group.source_name.as_str() };
            output_projection.push(JoinOutputColumn::new(
                side,
                ColumnRef::input(join_input_index, source_name, group.output_name.as_str()),
                group.output_name.as_str(),
            ));
        }
        for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
            output_projection.push(JoinOutputColumn::new(
                preagg_side,
                ColumnRef::input(join_side_index(preagg_side), state.alias.as_str(), state.alias.as_str()),
                state.alias.as_str(),
            ));
        }
        let spec = JoinSpec {
            join_type,
            keys,
            residual_predicate: None,
            output_projection,
            null_equality: join.spec.null_equality,
            build_side: join.spec.build_side,
            build_artifact_id: None,
            mark_column_name: None,
        };
        JoinNode { inputs, spec }
    }

    fn try_rewrite_preaggregate_through_join_union(
        &mut self,
        preagg_input: NodeId,
        preagg_spec: &AggregateSpec,
        preagg_schema: &SchemaRef,
    ) -> Option<NodeId> {
        if preagg_spec.grouping_sets.is_some()
            || preagg_spec.aggr_exprs.is_empty()
            || !preagg_spec.aggr_exprs.iter().all(decomposable_union_partial_aggregate_expr)
        {
            return None;
        }

        let input_node = self.nodes.get(preagg_input.index()).filter(|node| node.id == preagg_input)?;
        let PlanNode::Join(input_join) = &input_node.kind else {
            return None;
        };
        if input_join.spec.join_type != JoinType::Inner
            || input_join.spec.residual_predicate.is_some()
            || input_join.spec.build_artifact_id.is_some()
            || input_join.spec.mark_column_name.is_some()
        {
            return None;
        }

        let (union_side, union_inputs) = self.distributable_join_union_input(input_join)?;
        let other_side = opposite_side(union_side);
        let other_input = input_join.inputs[join_side_index(other_side)];
        let other_bytes = self.estimated_rewritten_output_bytes(other_input)?;
        let known_union_bytes = union_inputs
            .iter()
            .try_fold(0_u64, |sum, input| sum.checked_add(self.estimated_rewritten_output_bytes(*input)?))?;
        if !AggregateJoinPushdownPolicy::default().join_union_preaggregate_admitted(
            known_union_bytes,
            other_bytes,
            union_inputs.len(),
        ) {
            return None;
        }

        let input_join = input_join.clone();
        let join_facts = input_node.facts.clone();
        let branch_aggregates = union_inputs
            .into_iter()
            .map(|branch_input| {
                let branch_join = join_with_replaced_input(&input_join, union_side, branch_input);
                let branch_join_id = self.push_node(PlanNode::Join(branch_join), join_facts.clone());
                self.push_node(
                    PlanNode::Aggregate(AggregateNode { input: branch_join_id, spec: preagg_spec.clone() }),
                    PlanFacts::device(Arc::clone(preagg_schema), SchemaBehavior::Aggregate),
                )
            })
            .collect::<Vec<_>>();
        let partial_union = self.push_node(
            PlanNode::Union(crate::plan::UnionNode { inputs: branch_aggregates }),
            PlanFacts::device(Arc::clone(preagg_schema), SchemaBehavior::Union),
        );
        let final_spec = final_aggregate_spec_after_union_preagg(preagg_spec, preagg_schema)?;
        Some(self.push_node(
            PlanNode::Aggregate(AggregateNode { input: partial_union, spec: final_spec }),
            PlanFacts::device(Arc::clone(preagg_schema), SchemaBehavior::Aggregate),
        ))
    }

    fn distributable_join_union_input(&self, join: &JoinNode) -> Option<(JoinSide, Vec<NodeId>)> {
        [JoinSide::Left, JoinSide::Right].into_iter().find_map(|side| {
            let input = join.inputs[join_side_index(side)];
            let node = self.nodes.get(input.index()).filter(|node| node.id == input)?;
            let PlanNode::Union(union) = &node.kind else {
                return None;
            };
            (union.inputs.len() > 1).then(|| (side, union.inputs.clone()))
        })
    }

    fn estimated_rewritten_output_bytes(&self, node_id: NodeId) -> Option<u64> {
        let node = self.nodes.get(node_id.index()).filter(|node| node.id == node_id)?;
        match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                crate::source::SourceNode::LocalParquet(source) => source
                    .diagnostics
                    .selected_row_group_stats
                    .as_ref()
                    .and_then(|stats| stats.selected_source_bytes())
                    .or(source.diagnostics.known_bytes),
                crate::source::SourceNode::GpuResident(_) => None,
                crate::source::SourceNode::LiteralBatch(source) => source
                    .batches
                    .iter()
                    .try_fold(0_u64, |sum, batch| sum.checked_add(batch.get_array_memory_size() as u64)),
            },
            PlanNode::Projection(projection) => self.estimated_rewritten_output_bytes(projection.input),
            PlanNode::Filter(filter) => self.estimated_rewritten_output_bytes(filter.input),
            PlanNode::Sort(sort) => self.estimated_rewritten_output_bytes(sort.input),
            PlanNode::Union(union) => union
                .inputs
                .iter()
                .try_fold(0_u64, |total, input| total.checked_add(self.estimated_rewritten_output_bytes(*input)?)),
            PlanNode::Limit(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => None,
        }
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        if let Some(existing) = self.nodes.iter().find(|node| node.kind == kind && node.facts == facts) {
            return existing.id;
        }
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

#[derive(Clone)]
struct LineageColumn {
    side: JoinSide,
    source_name: String,
    output_name: String,
}

/// Lineage entry for one aggregate measure expression after the projection
/// chain between the aggregate and the join has been inlined.
#[derive(Clone)]
struct LineageMeasureColumn {
    side: JoinSide,
    /// Aggregate measure expression rewritten against the join's `side`-input
    /// column names. This is the expression the pre-aggregate evaluates.
    side_input_aggregate: AggregateExpr,
}

#[derive(Clone)]
struct PreAggregateColumn {
    source_name: String,
    output_name: String,
}

#[derive(Clone)]
struct PreAggregateMeasureState {
    original_function: AggregateFunction,
    original_alias: String,
    final_output_field: Field,
    state_columns: Vec<PreAggregateStateColumn>,
}

#[derive(Clone)]
struct PreAggregateStateColumn {
    alias: String,
    preaggregate_function: AggregateFunction,
    field: Field,
}

struct FinalAggregateRewrite {
    spec: AggregateSpec,
    output_schema: SchemaRef,
    projection_outputs: Option<Vec<crate::plan::ProjectionOutput>>,
}

struct AggregateJoinLineage {
    group_columns: Vec<LineageColumn>,
    measure_columns: Vec<LineageMeasureColumn>,
}

impl AggregateJoinLineage {
    fn new(
        aggregate: &AggregateNode,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<Self, AggregateJoinPushdownRejectedReason> {
        let group_columns = aggregate
            .spec
            .group_exprs
            .iter()
            .map(|expr| group_lineage_column(expr, join, projection_chain))
            .collect::<Result<Vec<_>, _>>()?;
        let measure_columns = aggregate
            .spec
            .aggr_exprs
            .iter()
            .map(|expr| measure_lineage_column(expr, join, projection_chain))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { group_columns, measure_columns })
    }

    fn single_group_side(&self) -> Result<JoinSide, AggregateJoinPushdownRejectedReason> {
        single_side(self.group_columns.iter().map(|column| column.side))
    }

    fn single_measure_side(&self) -> Result<JoinSide, AggregateJoinPushdownRejectedReason> {
        single_side(self.measure_columns.iter().map(|column| column.side))
    }

    fn group_columns_for_side(&self, side: JoinSide) -> Vec<PreAggregateColumn> {
        self.group_columns
            .iter()
            .filter(|column| column.side == side)
            .map(|column| PreAggregateColumn {
                source_name: column.source_name.clone(),
                output_name: column.output_name.clone(),
            })
            .collect()
    }
}

fn single_side(mut sides: impl Iterator<Item = JoinSide>) -> Result<JoinSide, AggregateJoinPushdownRejectedReason> {
    let first = sides.next().ok_or(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern)?;
    if sides.any(|side| side != first) {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
    }
    Ok(first)
}

fn group_lineage_column(
    expr: &Expr,
    join: &JoinNode,
    chain: &[ProjectionNode],
) -> Result<LineageColumn, AggregateJoinPushdownRejectedReason> {
    // Group keys must surface as plain column references at the aggregate
    // boundary. The aggregate's view-name comes from the original group expr;
    // the inlined column resolves which join output column it actually reads.
    let aggregate_view_column =
        simple_projection_column(expr).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
    let aggregate_output_name = aggregate_view_column.name().to_string();
    let inlined = inline_expr_through_projection_chain(expr.clone(), chain)
        .map_err(|_| AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
    let join_output_column =
        simple_projection_column(&inlined).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
    let output = unique_join_output_for_name(join, join_output_column.name())
        .or_else(|| unique_join_output_for_name(join, join_output_column.display_name()))
        .ok_or(AggregateJoinPushdownRejectedReason::MissingJoinOutputLineage)?;
    Ok(LineageColumn {
        side: output.side,
        source_name: output.column.name().to_string(),
        output_name: aggregate_output_name,
    })
}

fn measure_lineage_column(
    expr: &AggregateExpr,
    join: &JoinNode,
    chain: &[ProjectionNode],
) -> Result<LineageMeasureColumn, AggregateJoinPushdownRejectedReason> {
    if !matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Count | AggregateFunction::Avg)
        || expr.distinct
        || expr.filter.is_some()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let inlined = inline_expr_through_projection_chain(expr.expr.clone(), chain)
        .map_err(|_| AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
    let (side, side_input_expr) = resolve_expr_side_and_remap_to_input(&inlined, join)?;
    Ok(LineageMeasureColumn {
        side,
        side_input_aggregate: AggregateExpr {
            function: expr.function,
            expr: side_input_expr,
            alias: expr.alias.clone(),
            distinct: expr.distinct,
            filter: expr.filter.clone(),
        },
    })
}

/// Inlines an expression through a chain of projection nodes. `chain[0]` is the
/// outermost projection (closest to the aggregate) and `chain[last]` is the
/// innermost (closest to the join). After inlining, the returned expression is
/// in terms of the join's output column names.
fn inline_expr_through_projection_chain(
    expr: Expr,
    chain: &[ProjectionNode],
) -> Result<Expr, AggregateJoinPushdownRejectedReason> {
    let mut current = expr;
    for projection in chain {
        current = substitute_columns_through_projection(current, projection)?;
    }
    Ok(current)
}

fn substitute_columns_through_projection(
    expr: Expr,
    projection: &ProjectionNode,
) -> Result<Expr, AggregateJoinPushdownRejectedReason> {
    let mut map: BTreeMap<&str, &Expr> = BTreeMap::new();
    for output in &projection.outputs {
        if map.insert(output.field.name(), &output.expr).is_some() {
            // Two outputs share the same name — ambiguous lineage.
            return Err(AggregateJoinPushdownRejectedReason::MissingJoinOutputLineage);
        }
    }
    Ok(substitute_unqualified_columns(expr, &|name| map.get(name).map(|expr| peel_outer_alias((**expr).clone()))))
}

fn peel_outer_alias(expr: Expr) -> Expr {
    match expr {
        Expr::Alias { expr, .. } => peel_outer_alias(*expr),
        other => other,
    }
}

/// Resolves which side of `join` an expression draws its columns from and
/// rewrites the expression in terms of that side's input column names.
///
/// Returns [`AggregateJoinPushdownRejectedReason::UnsupportedSidePattern`] when
/// the expression mixes columns from both sides of the join.
fn resolve_expr_side_and_remap_to_input(
    expr: &Expr,
    join: &JoinNode,
) -> Result<(JoinSide, Expr), AggregateJoinPushdownRejectedReason> {
    let mut names = Vec::new();
    collect_unqualified_column_names(expr, &mut names);
    if names.is_empty() {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let mut resolved_side: Option<JoinSide> = None;
    let mut name_to_input: BTreeMap<String, String> = BTreeMap::new();
    for name in &names {
        let output = unique_join_output_for_name(join, name.as_str())
            .ok_or(AggregateJoinPushdownRejectedReason::MissingJoinOutputLineage)?;
        if let Some(existing) = resolved_side
            && existing != output.side
        {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
        }
        resolved_side = Some(output.side);
        name_to_input.insert(name.clone(), output.column.name().to_string());
    }
    let side = resolved_side.expect("at least one column present");
    let remapped = substitute_unqualified_columns(expr.clone(), &|name| {
        name_to_input.get(name).map(|input_name| Expr::column(input_name.clone()))
    });
    Ok((side, remapped))
}

fn collect_unqualified_column_names(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Column(column) => {
            if column.qualifier().is_none() && !out.iter().any(|n| n == column.name()) {
                out.push(column.name().to_string());
            }
        },
        Expr::Literal(_) => {},
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::Abs(inner) => collect_unqualified_column_names(inner, out),
        Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. } => collect_unqualified_column_names(expr, out),
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_unqualified_column_names(left, out);
            collect_unqualified_column_names(right, out);
        },
        Expr::Concat(parts) => {
            for part in parts {
                collect_unqualified_column_names(part, out);
            }
        },
        Expr::InList { expr, list, .. } => {
            collect_unqualified_column_names(expr, out);
            for item in list {
                collect_unqualified_column_names(item, out);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_unqualified_column_names(&branch.when, out);
                collect_unqualified_column_names(&branch.then, out);
            }
            collect_unqualified_column_names(else_expr, out);
        },
    }
}

fn substitute_unqualified_columns(expr: Expr, map: &dyn Fn(&str) -> Option<Expr>) -> Expr {
    match expr {
        Expr::Column(column) => {
            if column.qualifier().is_none()
                && let Some(replacement) = map(column.name())
            {
                return replacement;
            }
            Expr::Column(column)
        },
        Expr::Literal(_) => expr,
        Expr::Not(inner) => Expr::Not(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::IsNull(inner) => Expr::IsNull(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::IsNotNull(inner) => Expr::IsNotNull(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::Cast { expr, target_type } => {
            Expr::Cast { expr: Box::new(substitute_unqualified_columns(*expr, map)), target_type }
        },
        Expr::TryCast { expr, target_type } => {
            Expr::TryCast { expr: Box::new(substitute_unqualified_columns(*expr, map)), target_type }
        },
        Expr::Binary { left, op, right, output_type } => Expr::Binary {
            left: Box::new(substitute_unqualified_columns(*left, map)),
            op,
            right: Box::new(substitute_unqualified_columns(*right, map)),
            output_type,
        },
        Expr::Like { expr, pattern, escape, negated } => {
            Expr::Like { expr: Box::new(substitute_unqualified_columns(*expr, map)), pattern, escape, negated }
        },
        Expr::Substring { expr, start, length } => {
            Expr::Substring { expr: Box::new(substitute_unqualified_columns(*expr, map)), start, length }
        },
        Expr::Lower(inner) => Expr::Lower(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::Upper(inner) => Expr::Upper(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::StringLength { expr, output_type } => {
            Expr::StringLength { expr: Box::new(substitute_unqualified_columns(*expr, map)), output_type }
        },
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => Expr::RegexpReplace {
            expr: Box::new(substitute_unqualified_columns(*expr, map)),
            pattern,
            replacement,
            output_type,
        },
        Expr::Concat(parts) => {
            Expr::Concat(parts.into_iter().map(|part| substitute_unqualified_columns(part, map)).collect())
        },
        Expr::Abs(inner) => Expr::Abs(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::Sqrt { expr, output_type } => {
            Expr::Sqrt { expr: Box::new(substitute_unqualified_columns(*expr, map)), output_type }
        },
        Expr::Round { expr, decimal_places, output_type } => {
            Expr::Round { expr: Box::new(substitute_unqualified_columns(*expr, map)), decimal_places, output_type }
        },
        Expr::InList { expr, list, negated } => Expr::InList {
            expr: Box::new(substitute_unqualified_columns(*expr, map)),
            list: list.into_iter().map(|item| substitute_unqualified_columns(item, map)).collect(),
            negated,
        },
        Expr::IsDistinctFrom { left, right } => Expr::IsDistinctFrom {
            left: Box::new(substitute_unqualified_columns(*left, map)),
            right: Box::new(substitute_unqualified_columns(*right, map)),
        },
        Expr::IsNotDistinctFrom { left, right } => Expr::IsNotDistinctFrom {
            left: Box::new(substitute_unqualified_columns(*left, map)),
            right: Box::new(substitute_unqualified_columns(*right, map)),
        },
        Expr::Case { branches, else_expr } => Expr::Case {
            branches: branches
                .into_iter()
                .map(|branch| CaseWhen {
                    when: substitute_unqualified_columns(branch.when, map),
                    then: substitute_unqualified_columns(branch.then, map),
                })
                .collect(),
            else_expr: Box::new(substitute_unqualified_columns(*else_expr, map)),
        },
        Expr::Extract { field, expr } => {
            Expr::Extract { field, expr: Box::new(substitute_unqualified_columns(*expr, map)) }
        },
        Expr::DateTrunc { field, expr, output_type } => {
            Expr::DateTrunc { field, expr: Box::new(substitute_unqualified_columns(*expr, map)), output_type }
        },
        Expr::Alias { expr, alias } => substitute_unqualified_columns(*expr, map).alias(alias),
    }
}

fn push_unique_preagg_group_column(
    columns: &mut Vec<PreAggregateColumn>,
    source_name: &str,
    output_name: &str,
    input_schema: &SchemaRef,
) -> Result<(), AggregateJoinPushdownRejectedReason> {
    if let Some(existing) = columns.iter().find(|column| column.output_name == output_name) {
        if existing.source_name == source_name {
            return Ok(());
        }
        return Err(AggregateJoinPushdownRejectedReason::ConflictingPreAggregateOutputName);
    }
    input_schema.field_with_name(source_name).map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
    columns.push(PreAggregateColumn { source_name: source_name.to_string(), output_name: output_name.to_string() });
    Ok(())
}

fn preaggregate_schema(
    input_schema: &SchemaRef,
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    group_columns: &[PreAggregateColumn],
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<SchemaRef, AggregateJoinPushdownRejectedReason> {
    let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
    let mut fields = Vec::with_capacity(group_columns.len() + state_count);
    for column in group_columns {
        let input_field = input_schema
            .field_with_name(column.source_name.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        let mut field = input_field.as_ref().clone();
        if field.name() != &column.output_name {
            field = Field::new(column.output_name.as_str(), field.data_type().clone(), field.is_nullable());
        }
        fields.push(field);
    }
    if aggregate.spec.aggr_exprs.len() != preagg_measure_states.len()
        || preagg_measure_states
            .iter()
            .any(|measure| aggregate_output_schema.field_with_name(measure.original_alias.as_str()).is_err())
    {
        return Err(AggregateJoinPushdownRejectedReason::MissingInputField);
    }
    for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
        fields.push(state.field.clone());
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn preaggregate_spec(
    aggregate: &AggregateNode,
    group_columns: &[PreAggregateColumn],
    preagg_measure_states: &[PreAggregateMeasureState],
    measure_input_aliases: &[String],
) -> Result<AggregateSpec, AggregateJoinPushdownRejectedReason> {
    // Both group keys and measure inputs are read from the materialized
    // measure-input projection above the join's measure-side input. The
    // projection preserves group/key column names by source name, so the
    // pre-aggregate's group_exprs still alias from source name to output name.
    let group_exprs = group_columns
        .iter()
        .map(|column| Expr::column(column.source_name.as_str()).alias(column.output_name.as_str()))
        .collect::<Vec<_>>();
    if measure_input_aliases.len() != preagg_measure_states.len()
        || aggregate.spec.aggr_exprs.len() != preagg_measure_states.len()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let aggr_exprs = measure_input_aliases
        .iter()
        .zip(preagg_measure_states)
        .flat_map(|(measure_input_alias, measure)| {
            measure.state_columns.iter().map(|state| {
                AggregateExpr::new(
                    state.preaggregate_function,
                    Expr::column(measure_input_alias.as_str()),
                    state.alias.as_str(),
                )
            })
        })
        .collect::<Vec<_>>();
    Ok(AggregateSpec::new(group_exprs, aggr_exprs))
}

fn preaggregate_measure_states(
    input_schema: &SchemaRef,
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    state_aliases: &[Vec<String>],
    measure_input_aliases: &[String],
) -> Result<Vec<PreAggregateMeasureState>, AggregateJoinPushdownRejectedReason> {
    if aggregate.spec.aggr_exprs.len() != state_aliases.len()
        || aggregate.spec.aggr_exprs.len() != measure_input_aliases.len()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    aggregate
        .spec
        .aggr_exprs
        .iter()
        .zip(state_aliases)
        .zip(measure_input_aliases)
        .map(|((expr, aliases), measure_input_alias)| {
            let output_field = aggregate_output_schema
                .field_with_name(expr.alias.as_str())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
                .as_ref()
                .clone();
            let input_field = input_schema
                .field_with_name(measure_input_alias.as_str())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
            let state_columns = match expr.function {
                AggregateFunction::Sum | AggregateFunction::Count => {
                    let alias =
                        aliases.first().ok_or(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
                    vec![PreAggregateStateColumn {
                        alias: alias.clone(),
                        preaggregate_function: expr.function,
                        field: Field::new(alias.as_str(), output_field.data_type().clone(), output_field.is_nullable()),
                    }]
                },
                AggregateFunction::Avg => {
                    if aliases.len() != 2 {
                        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
                    }
                    let sum_alias = aliases[0].clone();
                    let count_alias = aliases[1].clone();
                    let sum_type = avg_pushdown_sum_state_data_type(input_field.data_type(), output_field.data_type())?;
                    vec![
                        PreAggregateStateColumn {
                            alias: sum_alias.clone(),
                            preaggregate_function: AggregateFunction::Sum,
                            field: Field::new(sum_alias.as_str(), sum_type, true),
                        },
                        PreAggregateStateColumn {
                            alias: count_alias.clone(),
                            preaggregate_function: AggregateFunction::Count,
                            field: Field::new(count_alias.as_str(), DataType::Int64, false),
                        },
                    ]
                },
                _ => return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
            };
            Ok(PreAggregateMeasureState {
                original_function: expr.function,
                original_alias: expr.alias.clone(),
                final_output_field: output_field,
                state_columns,
            })
        })
        .collect()
}

/// Builds the projection outputs for the materializing projection inserted
/// above the join's measure-side input. Group/key columns are passed through
/// by source name; each measure expression is materialized once under a fresh
/// alias so the pre-aggregate that consumes the projection can reference it
/// as a plain column.
fn measure_input_projection_outputs(
    measure_input_schema: &SchemaRef,
    aggregate_input_schema: &SchemaRef,
    aggregate: &AggregateNode,
    group_columns: &[PreAggregateColumn],
    lineage: &AggregateJoinLineage,
    measure_input_aliases: &[String],
) -> Result<Vec<crate::plan::ProjectionOutput>, AggregateJoinPushdownRejectedReason> {
    if lineage.measure_columns.len() != measure_input_aliases.len()
        || aggregate.spec.aggr_exprs.len() != measure_input_aliases.len()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let mut outputs = Vec::with_capacity(group_columns.len() + measure_input_aliases.len());
    for column in group_columns {
        let field = measure_input_schema
            .field_with_name(column.source_name.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
            .clone();
        outputs.push(crate::plan::ProjectionOutput::new(Expr::column(column.source_name.as_str()), field));
    }
    for ((measure, alias), aggr_expr) in
        lineage.measure_columns.iter().zip(measure_input_aliases).zip(aggregate.spec.aggr_exprs.iter())
    {
        // Resolve the measure-input value type from the aggregate's input
        // schema, using the original aggr-expr column reference. The
        // projection chain preserved this name at the aggregate boundary, so
        // its arrow type matches what the inlined side-input expression
        // evaluates to.
        let original_column = simple_projection_column(&aggr_expr.expr)
            .ok_or(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
        let aggregate_input_field = aggregate_input_schema
            .field_with_name(original_column.name())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        let field =
            Field::new(alias.as_str(), aggregate_input_field.data_type().clone(), aggregate_input_field.is_nullable());
        outputs.push(crate::plan::ProjectionOutput::new(
            measure.side_input_aggregate.expr.clone().alias(alias.as_str()),
            field,
        ));
    }
    Ok(outputs)
}

fn unique_measure_input_alias(
    index: usize,
    aggregate_output_schema: &Schema,
    group_columns: &[PreAggregateColumn],
    preagg_aliases: &[String],
) -> String {
    let mut used = aggregate_output_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    used.extend(group_columns.iter().map(|column| column.output_name.clone()));
    used.extend(group_columns.iter().map(|column| column.source_name.clone()));
    used.extend(preagg_aliases.iter().cloned());
    for suffix in 0usize.. {
        let candidate = if suffix == 0 {
            format!("__cudf_preagg_arg_{index}")
        } else {
            format!("__cudf_preagg_arg_{index}_{suffix}")
        };
        if !used.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("infinite suffix search must find a measure-input alias")
}

fn final_aggregate_rewrite_after_preagg(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<FinalAggregateRewrite, AggregateJoinPushdownRejectedReason> {
    if aggregate.spec.aggr_exprs.len() != preagg_measure_states.len() {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let mut fields = Vec::with_capacity(
        aggregate.spec.group_exprs.len()
            + preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>(),
    );
    for group in &aggregate.spec.group_exprs {
        let column =
            simple_projection_column(group).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
        fields.push(
            aggregate_output_schema
                .field_with_name(column.name())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
                .as_ref()
                .clone(),
        );
    }
    if let Some(grouping_sets) = &aggregate.spec.grouping_sets {
        fields.push(
            aggregate_output_schema
                .field_with_name(grouping_sets.grouping_id_alias.as_str())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
                .as_ref()
                .clone(),
        );
    }

    let mut aggr_exprs = Vec::new();
    for measure in preagg_measure_states {
        match measure.original_function {
            AggregateFunction::Sum | AggregateFunction::Count => {
                let state = measure
                    .state_columns
                    .first()
                    .ok_or(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
                aggr_exprs.push(AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(state.alias.as_str()),
                    measure.original_alias.as_str(),
                ));
                fields.push(measure.final_output_field.clone());
            },
            AggregateFunction::Avg => {
                if measure.state_columns.len() != 2 {
                    return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
                }
                for state in &measure.state_columns {
                    aggr_exprs.push(AggregateExpr::new(
                        AggregateFunction::Sum,
                        Expr::column(state.alias.as_str()),
                        state.alias.as_str(),
                    ));
                    fields.push(state.field.clone());
                }
            },
            _ => return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        }
    }

    let spec = AggregateSpec {
        group_exprs: aggregate.spec.group_exprs.clone(),
        aggr_exprs,
        grouping_sets: aggregate.spec.grouping_sets.clone(),
    };
    let output_schema = Arc::new(Schema::new(fields));
    let projection_outputs = preagg_measure_states
        .iter()
        .any(|measure| measure.original_function == AggregateFunction::Avg)
        .then(|| final_avg_projection_outputs(aggregate, aggregate_output_schema, preagg_measure_states))
        .transpose()?;
    Ok(FinalAggregateRewrite { spec, output_schema, projection_outputs })
}

fn final_avg_projection_outputs(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<Vec<ProjectionOutput>, AggregateJoinPushdownRejectedReason> {
    let grouping_id_output_count = usize::from(aggregate.spec.grouping_sets.is_some());
    let mut outputs = Vec::with_capacity(
        aggregate.spec.group_exprs.len() + grouping_id_output_count + aggregate.spec.aggr_exprs.len(),
    );
    append_group_projection_outputs(&mut outputs, aggregate, aggregate_output_schema)?;
    if let Some(grouping_sets) = &aggregate.spec.grouping_sets {
        let field = aggregate_output_schema
            .field_with_name(grouping_sets.grouping_id_alias.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
            .as_ref()
            .clone();
        outputs.push(ProjectionOutput::new(Expr::column(grouping_sets.grouping_id_alias.as_str()), field));
    }

    for measure in preagg_measure_states {
        let field = measure.final_output_field.clone();
        let expr = match measure.original_function {
            AggregateFunction::Sum | AggregateFunction::Count => Expr::column(measure.original_alias.as_str()),
            AggregateFunction::Avg => final_avg_projection_expr(measure)?,
            _ => return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        };
        outputs.push(ProjectionOutput::new(expr, field));
    }
    Ok(outputs)
}

fn final_avg_projection_expr(measure: &PreAggregateMeasureState) -> Result<Expr, AggregateJoinPushdownRejectedReason> {
    if measure.state_columns.len() != 2 {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let sum_state = &measure.state_columns[0];
    let count_state = &measure.state_columns[1];
    let division_type =
        avg_pushdown_division_data_type(sum_state.field.data_type(), measure.final_output_field.data_type())?;
    let sum = Expr::cast(Expr::column(sum_state.alias.as_str()), division_type.clone());
    let count =
        Expr::cast(Expr::column(count_state.alias.as_str()), avg_pushdown_count_projection_data_type(&division_type));
    let count_is_zero =
        Expr::binary(Expr::column(count_state.alias.as_str()), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(0)));
    let avg_divide = if matches!(division_type, DataType::Decimal128(_, _)) {
        Expr::binary_with_output_type(sum, BinaryOp::Divide, count, division_type)
    } else {
        Expr::binary(sum, BinaryOp::Divide, count)
    };
    let avg = Expr::cast(avg_divide, measure.final_output_field.data_type().clone());
    Ok(Expr::case(
        vec![CaseWhen::new(
            count_is_zero,
            Expr::literal(LiteralValue::Null(measure.final_output_field.data_type().clone())),
        )],
        avg,
    )
    .alias(measure.original_alias.as_str()))
}

fn avg_pushdown_sum_state_data_type(
    input_type: &DataType,
    avg_output_type: &DataType,
) -> Result<DataType, AggregateJoinPushdownRejectedReason> {
    match avg_output_type {
        DataType::Float32 | DataType::Float64 => Ok(DataType::Float64),
        DataType::Decimal128(_, output_scale) => match input_type {
            DataType::Decimal128(_, input_scale) => {
                Ok(DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *input_scale))
            },
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => Ok(DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *output_scale)),
            _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        },
        _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
    }
}

fn avg_pushdown_division_data_type(
    state_sum_type: &DataType,
    avg_output_type: &DataType,
) -> Result<DataType, AggregateJoinPushdownRejectedReason> {
    match avg_output_type {
        DataType::Float32 | DataType::Float64 => Ok(DataType::Float64),
        DataType::Decimal128(_, output_scale) => match state_sum_type {
            DataType::Decimal128(_, _)
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64 => Ok(DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, *output_scale)),
            _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
        },
        _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
    }
}

fn avg_pushdown_count_projection_data_type(avg_output_type: &DataType) -> DataType {
    match avg_output_type {
        DataType::Decimal128(_, _) => DataType::Decimal128(arrow::datatypes::DECIMAL128_MAX_PRECISION, 0),
        _ => avg_output_type.clone(),
    }
}

fn decomposable_union_partial_aggregate_expr(expr: &AggregateExpr) -> bool {
    matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Count)
        && !expr.distinct
        && expr.filter.is_none()
}

fn join_with_replaced_input(join: &JoinNode, side: JoinSide, replacement: NodeId) -> JoinNode {
    let mut rewritten = join.clone();
    rewritten.inputs[join_side_index(side)] = replacement;
    rewritten
}

fn final_aggregate_spec_after_union_preagg(
    preagg_spec: &AggregateSpec,
    preagg_schema: &SchemaRef,
) -> Option<AggregateSpec> {
    if preagg_spec.grouping_sets.is_some() || preagg_schema.fields().len() < preagg_spec.group_exprs.len() {
        return None;
    }
    let group_exprs = preagg_schema
        .fields()
        .iter()
        .take(preagg_spec.group_exprs.len())
        .map(|field| Expr::column(field.name()))
        .collect::<Vec<_>>();
    let aggr_exprs = preagg_spec
        .aggr_exprs
        .iter()
        .map(|expr| AggregateExpr::new(AggregateFunction::Sum, Expr::column(expr.alias.as_str()), expr.alias.as_str()))
        .collect::<Vec<_>>();
    Some(AggregateSpec::new(group_exprs, aggr_exprs))
}

fn new_join_output_schema(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_schema: &SchemaRef,
    preagg_measure_states: &[PreAggregateMeasureState],
    preagg_side: JoinSide,
    join_type: JoinType,
) -> Result<SchemaRef, AggregateJoinPushdownRejectedReason> {
    let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
    let mut fields = Vec::with_capacity(aggregate.spec.group_exprs.len() + state_count);
    for group in &aggregate.spec.group_exprs {
        let column =
            simple_projection_column(group).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
        let field = aggregate_output_schema
            .field_with_name(column.name())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        fields.push(field.as_ref().clone());
    }
    for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
        let field = preagg_schema
            .field_with_name(state.alias.as_str())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
        let mut field = field.as_ref().clone();
        if join_side_introduces_nulls(join_type, preagg_side) && !field.is_nullable() {
            field = Field::new(field.name().as_str(), field.data_type().clone(), true);
        }
        fields.push(field);
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn coalesce_preagg_projection_outputs(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    preagg_measure_states: &[PreAggregateMeasureState],
) -> Result<Vec<ProjectionOutput>, AggregateJoinPushdownRejectedReason> {
    let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
    let mut outputs = Vec::with_capacity(aggregate.spec.group_exprs.len() + state_count);
    append_group_projection_outputs(&mut outputs, aggregate, aggregate_output_schema)?;

    for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
        let zero = zero_literal_for_data_type(state.field.data_type())?;
        let field = Field::new(state.alias.as_str(), state.field.data_type().clone(), state.field.is_nullable());
        let coalesced = Expr::case(
            vec![CaseWhen::new(Expr::is_null(Expr::column(state.alias.as_str())), Expr::literal(zero))],
            Expr::column(state.alias.as_str()),
        )
        .alias(state.alias.as_str());
        outputs.push(ProjectionOutput::new(coalesced, field));
    }
    Ok(outputs)
}

fn append_group_projection_outputs(
    outputs: &mut Vec<ProjectionOutput>,
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
) -> Result<(), AggregateJoinPushdownRejectedReason> {
    for group in &aggregate.spec.group_exprs {
        let column =
            simple_projection_column(group).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
        let field = aggregate_output_schema
            .field_with_name(column.name())
            .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?
            .as_ref()
            .clone();
        outputs.push(ProjectionOutput::new(Expr::column(column.name()), field));
    }
    Ok(())
}

fn zero_literal_for_data_type(
    data_type: &arrow::datatypes::DataType,
) -> Result<crate::expr::LiteralValue, AggregateJoinPushdownRejectedReason> {
    match data_type {
        arrow::datatypes::DataType::Int8 => Ok(crate::expr::LiteralValue::Int8(0)),
        arrow::datatypes::DataType::Int16 => Ok(crate::expr::LiteralValue::Int16(0)),
        arrow::datatypes::DataType::Int32 => Ok(crate::expr::LiteralValue::Int32(0)),
        arrow::datatypes::DataType::Int64 => Ok(crate::expr::LiteralValue::Int64(0)),
        arrow::datatypes::DataType::UInt8 => Ok(crate::expr::LiteralValue::UInt8(0)),
        arrow::datatypes::DataType::UInt16 => Ok(crate::expr::LiteralValue::UInt16(0)),
        arrow::datatypes::DataType::UInt32 => Ok(crate::expr::LiteralValue::UInt32(0)),
        arrow::datatypes::DataType::UInt64 => Ok(crate::expr::LiteralValue::UInt64(0)),
        _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression),
    }
}

fn join_side_introduces_nulls(join_type: JoinType, side: JoinSide) -> bool {
    matches!((join_type, side), (JoinType::Left, JoinSide::Right) | (JoinType::Full, JoinSide::Left | JoinSide::Right))
}

fn preaggregate_state_aliases(
    aggregate: &AggregateNode,
    aggregate_output_schema: &Schema,
    group_columns: &[PreAggregateColumn],
) -> Vec<Vec<String>> {
    let mut used = aggregate_output_schema.fields().iter().map(|field| field.name().clone()).collect::<BTreeSet<_>>();
    used.extend(group_columns.iter().map(|column| column.output_name.clone()));
    used.extend(group_columns.iter().map(|column| column.source_name.clone()));
    aggregate
        .spec
        .aggr_exprs
        .iter()
        .enumerate()
        .map(|(index, expr)| match expr.function {
            AggregateFunction::Avg => vec![
                unique_preagg_state_alias(format!("__cudf_preagg_avg_{index}_sum"), &mut used),
                unique_preagg_state_alias(format!("__cudf_preagg_avg_{index}_count"), &mut used),
            ],
            _ => vec![unique_preagg_state_alias(format!("__cudf_preagg_sum_{index}"), &mut used)],
        })
        .collect()
}

fn unique_preagg_state_alias(base: String, used: &mut BTreeSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("infinite suffix search must find a preaggregate state alias")
}

fn rewritten_join_key_after_preagg(key: &JoinKey, preagg_side: JoinSide) -> JoinKey {
    match preagg_side {
        JoinSide::Left => JoinKey::new(
            ColumnRef::input(0, key.left.display_name(), key.left.display_name()),
            ColumnRef::input(1, key.right.name(), key.right.display_name()),
        ),
        JoinSide::Right => JoinKey::new(
            ColumnRef::input(0, key.left.name(), key.left.display_name()),
            ColumnRef::input(1, key.right.display_name(), key.right.display_name()),
        ),
    }
}

fn join_key_for_side(key: &JoinKey, side: JoinSide) -> &ColumnRef {
    match side {
        JoinSide::Left => &key.left,
        JoinSide::Right => &key.right,
    }
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
