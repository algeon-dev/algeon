use std::collections::BTreeMap;

use crate::{
    expr::{ColumnRef, Expr},
    plan::{
        JoinKey, JoinNode, JoinOutputColumn, JoinSide, JoinType, NodeId, PlanFacts, PlanNode, ProjectionNode,
        ProjectionOutput, QueryNode, QueryPlan, ResultSink, SchemaBehavior,
    },
};

pub const JOIN_FILTER_REASSOCIATION_RULE_NAME: &str = "join_filter_reassociation";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinFilterReassociationOptions {
    pub enabled: bool,
}

impl JoinFilterReassociationOptions {
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for JoinFilterReassociationOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinFilterReassociationDiagnostics {
    pub considered_count: usize,
    pub applied_count: usize,
    pub candidates: Vec<JoinFilterReassociationCandidateDiagnostics>,
}

impl JoinFilterReassociationDiagnostics {
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
pub struct JoinFilterReassociationCandidateDiagnostics {
    pub rule_name: &'static str,
    pub candidate_node_id: NodeId,
    pub crossed_join_node_id: Option<NodeId>,
    pub filtered_child_node_id: Option<NodeId>,
    pub filtering_node_id: Option<NodeId>,
    pub applied: bool,
    pub rejected_reason: Option<JoinFilterReassociationRejectedReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinFilterReassociationRejectedReason {
    UnsupportedParentJoinShape,
    NonPreservedSideProjection,
    UnsupportedPreservedInputShape,
    MissingKeyLineage,
    AmbiguousKeyLineage,
    UnsupportedCrossedJoinShape,
}

pub(crate) struct JoinFilterReassociationResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: JoinFilterReassociationDiagnostics,
}

pub(crate) fn optimize_join_filter_reassociation(
    plan: QueryPlan,
    options: &JoinFilterReassociationOptions,
) -> JoinFilterReassociationResult {
    if !options.enabled {
        return JoinFilterReassociationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: JoinFilterReassociationDiagnostics::default(),
        };
    }

    let mut rewriter = JoinFilterReassociationRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let diagnostics = rewriter.diagnostics;
        return JoinFilterReassociationResult { plan, change: super::NativeOptimizerChange::Unchanged, diagnostics };
    }
    let sink = rewrite_sink(plan.sink(), new_root);
    JoinFilterReassociationResult {
        plan: QueryPlan::from_rewritten_parts(rewriter.nodes, new_root, sink),
        change: super::NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    }
}

struct JoinFilterReassociationRewriter<'a> {
    original: &'a QueryPlan,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    diagnostics: JoinFilterReassociationDiagnostics,
    changed: bool,
}

impl<'a> JoinFilterReassociationRewriter<'a> {
    fn new(original: &'a QueryPlan) -> Self {
        Self {
            original,
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
        if let PlanNode::Join(join) = &old_node.kind
            && let Some(new_id) = self.try_reassociate_candidate(old_id, join)
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
        match kind {
            PlanNode::Source(source) => PlanNode::Source(source.clone()),
            PlanNode::Filter(filter) => {
                let mut filter = filter.clone();
                filter.input = self.rewrite_subtree(filter.input);
                PlanNode::Filter(filter)
            },
            PlanNode::Projection(projection) => {
                let mut projection = projection.clone();
                projection.input = self.rewrite_subtree(projection.input);
                PlanNode::Projection(projection)
            },
            PlanNode::Aggregate(aggregate) => {
                let mut aggregate = aggregate.clone();
                aggregate.input = self.rewrite_subtree(aggregate.input);
                PlanNode::Aggregate(aggregate)
            },
            PlanNode::Sort(sort) => {
                let mut sort = sort.clone();
                sort.input = self.rewrite_subtree(sort.input);
                PlanNode::Sort(sort)
            },
            PlanNode::Limit(limit) => {
                let mut limit = limit.clone();
                limit.input = self.rewrite_subtree(limit.input);
                PlanNode::Limit(limit)
            },
            PlanNode::Join(join) => {
                let mut join = join.clone();
                join.inputs = [self.rewrite_subtree(join.inputs[0]), self.rewrite_subtree(join.inputs[1])];
                PlanNode::Join(join)
            },
        }
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

        match self.plan_reassociation(candidate_node_id, join) {
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

    fn plan_reassociation(
        &mut self,
        candidate_node_id: NodeId,
        join: &JoinNode,
    ) -> Result<ReassociationPlan, JoinFilterReassociationRejectedReason> {
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
            || crossed_join.spec.null_equality != crate::plan::JoinNullEquality::NullsDoNotMatch
        {
            return Err(JoinFilterReassociationRejectedReason::UnsupportedCrossedJoinShape);
        }

        let filtered_child_old = crossed_join.inputs[join_side_index(lineage.filtered_side)];
        let filtering_new = self.rewrite_subtree(filtering_input);
        let filtered_child_new = self.rewrite_subtree(filtered_child_old);
        let filter_join_new = self.build_filter_join(filtered_child_new, filtering_new, join, preserved_side, &lineage);

        let mut rebuilt_crossed = crossed_join.clone();
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

        let parent_node = self.original.node(candidate_node_id).expect("candidate node must exist");
        let parent_projection = parent_projection_for_preserved_side(join, preserved_side, parent_node.output_schema())
            .ok_or(JoinFilterReassociationRejectedReason::NonPreservedSideProjection)?;
        let new_id = self.push_node(
            PlanNode::Projection(ProjectionNode::new(preserved_new, parent_projection)),
            PlanFacts::device(parent_node.output_schema().clone(), SchemaBehavior::Projection),
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
        let filtered_schema = self.nodes[filtered_child_new.index()].output_schema().clone();
        let projection = filtered_schema
            .fields()
            .iter()
            .map(|field| JoinOutputColumn::left(ColumnRef::input(0, field.name(), field.name()), field.name()))
            .collect::<Vec<_>>();
        let spec = crate::plan::JoinSpec::new(JoinType::Inner, keys)
            .with_output_projection(projection)
            .with_null_equality(crate::plan::JoinNullEquality::NullsDoNotMatch);
        self.push_node(
            PlanNode::Join(JoinNode { inputs: [filtered_child_new, filtering_new], spec }),
            PlanFacts::device(filtered_schema, SchemaBehavior::Join),
        )
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

struct ReassociationPlan {
    new_id: NodeId,
    crossed_join_node_id: NodeId,
    filtered_child_node_id: NodeId,
    filtering_node_id: NodeId,
}

struct PreservedInputLineage {
    projection_node_id: Option<NodeId>,
    crossed_join_node_id: NodeId,
    filtered_side: JoinSide,
    filtered_child_columns: Vec<ColumnRef>,
}

impl PreservedInputLineage {
    fn analyze(
        plan: &QueryPlan,
        preserved_input: NodeId,
        required_columns: &[ColumnRef],
    ) -> Result<Self, JoinFilterReassociationRejectedReason> {
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
                    .collect::<Result<Vec<_>, _>>()?;
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
    ) -> Result<Self, JoinFilterReassociationRejectedReason> {
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
        && join.spec.null_equality == crate::plan::JoinNullEquality::NullsDoNotMatch
}

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
    output_schema: &arrow::datatypes::SchemaRef,
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

fn simple_projection_column(expr: &Expr) -> Option<&ColumnRef> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias { expr, .. } => simple_projection_column(expr),
        _ => None,
    }
}

fn unique_join_output_for_name<'a>(join: &'a JoinNode, output_name: &str) -> Option<&'a JoinOutputColumn> {
    let mut matches = join.spec.output_projection.iter().filter(|output| output.output_name == output_name);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn rewrite_sink(sink: &ResultSink, root: NodeId) -> ResultSink {
    match sink {
        ResultSink::GpuResident { .. } => ResultSink::GpuResident { input: root },
        ResultSink::HostArrow { schema, .. } => ResultSink::HostArrow { input: root, schema: schema.clone() },
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
