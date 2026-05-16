#![allow(dead_code)]

use std::collections::BTreeMap;

use arrow::datatypes::{Field, Schema};

use crate::{
    adaptive_report::AdaptiveCandidateStatus,
    expr::{ColumnQualifier, ColumnRef, Expr},
    plan::{
        JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinType, NodeId, PlanFacts, PlanNode,
        ProjectionOutput, QueryNode, QueryPlan, ResultSink, SchemaBehavior, estimated_output_bytes,
        estimated_output_rows,
    },
};

pub const SEMIJOIN_REDUCTION_RULE_NAME: &str = "semijoin_reduction";
pub const DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER: u64 = 8;

/// Configuration for the semijoin-reduction optimizer pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionOptions {
    pub enabled: bool,
    pub cost_threshold_multiplier: u64,
    pub admission_mode: SemijoinReductionAdmissionMode,
}

impl SemijoinReductionOptions {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            cost_threshold_multiplier: DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER,
            admission_mode: SemijoinReductionAdmissionMode::NotEvaluated,
        }
    }

    pub fn with_admission_mode(mut self, admission_mode: SemijoinReductionAdmissionMode) -> Self {
        self.admission_mode = admission_mode;
        self
    }
}

impl Default for SemijoinReductionOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            cost_threshold_multiplier: DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER,
            admission_mode: SemijoinReductionAdmissionMode::NotEvaluated,
        }
    }
}

/// Mode-specific admission gate to use after a tentative semijoin rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemijoinReductionAdmissionMode {
    NotEvaluated,
    StrictBounded,
    Adaptive,
}

/// Result for one semijoin-reduction optimizer pass invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct SemijoinReductionResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: SemijoinReductionDiagnostics,
}

/// Diagnostics for semijoin-reduction candidates considered by the optimizer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SemijoinReductionDiagnostics {
    pub considered_count: usize,
    pub applied_count: usize,
    pub candidates: Vec<SemijoinReductionCandidateDiagnostics>,
}

impl SemijoinReductionDiagnostics {
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }
}

/// Per-candidate semijoin-reduction evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionCandidateDiagnostics {
    pub rule_name: &'static str,
    pub candidate_node_id: NodeId,
    pub before_root_node_id: NodeId,
    pub after_root_node_id: Option<NodeId>,
    pub crossed_join_node_ids: Vec<NodeId>,
    pub target_subtree_id: Option<NodeId>,
    pub key_mappings: Vec<SemijoinReductionKeyMappingDiagnostics>,
    pub cost_gate: Option<SemijoinReductionCostGateDiagnostics>,
    pub admission_decision: Option<SemijoinReductionAdmissionDecision>,
    pub original_admitted: Option<bool>,
    pub rewritten_admitted: Option<bool>,
    pub applied: bool,
    pub rejected_reason: Option<SemijoinReductionRejectedReason>,
}

/// One left-semi key mapping before and after a pushdown step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionKeyMappingDiagnostics {
    pub before_column: String,
    pub after_column: Option<String>,
    pub target_subtree_id: Option<NodeId>,
}

/// Cost-gate evidence used to decide whether a semijoin pushdown is worth keeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionCostGateDiagnostics {
    pub signal: SemijoinReductionCostSignal,
    pub threshold_multiplier: u64,
    pub accepted: bool,
}

/// Cost signal used by semijoin reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemijoinReductionCostSignal {
    RowEstimate { semi_right_rows: u64, target_rows: u64 },
    ByteEstimate { semi_right_bytes: u64, target_bytes: u64 },
    AggregateKeysetShape,
    Unavailable,
}

/// Mode-specific admission decision that caused a rewrite to be kept or rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemijoinReductionAdmissionDecision {
    NotEvaluated,
    StrictBoundedKept,
    StrictBoundedRejectedRewrite,
    AdaptiveKept,
    AdaptiveRejectedRewrite,
}

/// Generic semijoin-reduction rejection reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemijoinReductionRejectedReason {
    PassDisabled,
    NotLeftSemiJoin,
    UnsupportedLeftShape,
    UnsupportedCrossedJoinType,
    UnsupportedNullEquality,
    MissingKeyLineage,
    AmbiguousKeyLineage,
    ExpressionDerivedKey,
    DroppedKeyLineage,
    NonLeftOnlySemiProjection,
    CostGateRejected,
    RootSchemaMismatch,
    RootOutputProjectionMismatch,
    AdmissionRejectedRewrite,
}

/// Runs the semijoin-reduction pass.
pub fn optimize_semijoin_reduction(plan: QueryPlan, options: &SemijoinReductionOptions) -> SemijoinReductionResult {
    if !options.enabled {
        return SemijoinReductionResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: SemijoinReductionDiagnostics::default(),
        };
    }

    let mut rewriter = SemijoinReductionRewriter::new(&plan, options);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    let mut diagnostics = rewriter.diagnostics;

    if !rewriter.changed {
        return SemijoinReductionResult { plan, change: super::NativeOptimizerChange::Unchanged, diagnostics };
    }

    let sink = sink_with_input(plan.sink(), new_root);
    let rewritten_plan = QueryPlan::from_rewritten_parts(rewriter.nodes, new_root, sink);
    if let Err(reason) = validate_rewritten_root_semantics(&plan, &rewritten_plan) {
        reject_applied_candidates(&mut diagnostics, reason, None, None, None);
        return SemijoinReductionResult { plan, change: super::NativeOptimizerChange::Unchanged, diagnostics };
    }

    match evaluate_admission_gate(&plan, &rewritten_plan, options.admission_mode) {
        AdmissionGateDecision::Keep { decision, original_admitted, rewritten_admitted } => {
            record_admission_for_applied_candidates(&mut diagnostics, decision, original_admitted, rewritten_admitted);
            SemijoinReductionResult { plan: rewritten_plan, change: super::NativeOptimizerChange::Changed, diagnostics }
        },
        AdmissionGateDecision::Reject { decision, original_admitted, rewritten_admitted } => {
            reject_applied_candidates(
                &mut diagnostics,
                SemijoinReductionRejectedReason::AdmissionRejectedRewrite,
                Some(decision),
                original_admitted,
                rewritten_admitted,
            );
            SemijoinReductionResult { plan, change: super::NativeOptimizerChange::Unchanged, diagnostics }
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SemijoinLineageAnalysis {
    pub target_subtree_id: NodeId,
    pub target_key_columns: Vec<ColumnRef>,
    pub key_mappings: Vec<SemijoinReductionKeyMappingDiagnostics>,
    pub steps: Vec<SemijoinLineageStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SemijoinLineageStep {
    pub crossed_join_node_id: NodeId,
    pub target_side: JoinSide,
    pub target_child_id: NodeId,
    pub key_mappings: Vec<SemijoinReductionKeyMappingDiagnostics>,
}

struct SemijoinRewritePlan {
    new_id: NodeId,
    analysis: SemijoinLineageAnalysis,
    cost_gate: SemijoinReductionCostGateDiagnostics,
}

struct SemijoinRewriteRejection {
    reason: SemijoinReductionRejectedReason,
    analysis: Option<SemijoinLineageAnalysis>,
    cost_gate: Option<SemijoinReductionCostGateDiagnostics>,
}

impl SemijoinRewriteRejection {
    fn new(reason: SemijoinReductionRejectedReason) -> Self {
        Self { reason, analysis: None, cost_gate: None }
    }

    fn with_analysis(reason: SemijoinReductionRejectedReason, analysis: SemijoinLineageAnalysis) -> Self {
        Self { reason, analysis: Some(analysis), cost_gate: None }
    }
}

enum AdmissionGateDecision {
    Keep {
        decision: SemijoinReductionAdmissionDecision,
        original_admitted: Option<bool>,
        rewritten_admitted: Option<bool>,
    },
    Reject {
        decision: SemijoinReductionAdmissionDecision,
        original_admitted: Option<bool>,
        rewritten_admitted: Option<bool>,
    },
}

struct SemijoinLineageAnalyzer<'a> {
    plan: &'a QueryPlan,
}

impl<'a> SemijoinLineageAnalyzer<'a> {
    fn new(plan: &'a QueryPlan) -> Self {
        Self { plan }
    }

    fn analyze_left_semi_key_lineage(
        &self,
        left_input: NodeId,
        keys: &[JoinKey],
    ) -> Result<SemijoinLineageAnalysis, SemijoinReductionRejectedReason> {
        if keys.is_empty() {
            return Err(SemijoinReductionRejectedReason::MissingKeyLineage);
        }
        let required_columns = keys.iter().map(|key| key.left.clone()).collect::<Vec<_>>();
        self.analyze_required_columns(left_input, &required_columns)
    }

    fn analyze_required_columns(
        &self,
        node_id: NodeId,
        required_columns: &[ColumnRef],
    ) -> Result<SemijoinLineageAnalysis, SemijoinReductionRejectedReason> {
        let node = self.plan.node(node_id).ok_or(SemijoinReductionRejectedReason::MissingKeyLineage)?;
        match &node.kind {
            PlanNode::Source(_) => self.analyze_source(node_id, required_columns),
            PlanNode::Filter(filter) => self.analyze_required_columns(filter.input, required_columns),
            PlanNode::Projection(projection) => {
                let mut input_columns = Vec::with_capacity(required_columns.len());
                let mut local_mappings = Vec::with_capacity(required_columns.len());
                for column in required_columns {
                    let slot = resolve_output_slot(node.output_schema(), column)?;
                    let output =
                        projection.outputs.get(slot).ok_or(SemijoinReductionRejectedReason::DroppedKeyLineage)?;
                    let input_column = simple_projection_column(&output.expr)
                        .ok_or(SemijoinReductionRejectedReason::ExpressionDerivedKey)?;
                    local_mappings.push(SemijoinReductionKeyMappingDiagnostics {
                        before_column: column_label(column),
                        after_column: Some(column_label(input_column)),
                        target_subtree_id: Some(projection.input),
                    });
                    input_columns.push(input_column.clone());
                }
                let mut analysis = self.analyze_required_columns(projection.input, &input_columns)?;
                prepend_mappings(&mut analysis.key_mappings, local_mappings);
                Ok(analysis)
            },
            PlanNode::Join(join) => {
                if join.spec.join_type != JoinType::Inner {
                    return Err(SemijoinReductionRejectedReason::UnsupportedCrossedJoinType);
                }
                if join.spec.output_projection.is_empty() {
                    return Err(SemijoinReductionRejectedReason::DroppedKeyLineage);
                }

                let mut target_side = None;
                let mut child_columns = Vec::with_capacity(required_columns.len());
                let mut local_mappings = Vec::with_capacity(required_columns.len());

                for column in required_columns {
                    let slot = resolve_output_slot(node.output_schema(), column)?;
                    let output = join
                        .spec
                        .output_projection
                        .get(slot)
                        .ok_or(SemijoinReductionRejectedReason::DroppedKeyLineage)?;
                    validate_join_output_column_qualifier(output.side, &output.column)?;
                    match target_side {
                        Some(side) if side != output.side => {
                            return Err(SemijoinReductionRejectedReason::AmbiguousKeyLineage);
                        },
                        Some(_) => {},
                        None => target_side = Some(output.side),
                    }
                    let child_id = join.inputs[join_side_index(output.side)];
                    local_mappings.push(SemijoinReductionKeyMappingDiagnostics {
                        before_column: column_label(column),
                        after_column: Some(column_label(&output.column)),
                        target_subtree_id: Some(child_id),
                    });
                    child_columns.push(output.column.clone());
                }

                let target_side = target_side.ok_or(SemijoinReductionRejectedReason::MissingKeyLineage)?;
                let child_id = join.inputs[join_side_index(target_side)];
                let mut analysis = self.analyze_required_columns(child_id, &child_columns)?;
                analysis.steps.insert(
                    0,
                    SemijoinLineageStep {
                        crossed_join_node_id: node_id,
                        target_side,
                        target_child_id: child_id,
                        key_mappings: local_mappings.clone(),
                    },
                );
                prepend_mappings(&mut analysis.key_mappings, local_mappings);
                Ok(analysis)
            },
            PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => {
                Err(SemijoinReductionRejectedReason::UnsupportedLeftShape)
            },
        }
    }

    fn analyze_source(
        &self,
        node_id: NodeId,
        required_columns: &[ColumnRef],
    ) -> Result<SemijoinLineageAnalysis, SemijoinReductionRejectedReason> {
        let node = self.plan.node(node_id).ok_or(SemijoinReductionRejectedReason::MissingKeyLineage)?;
        let mut key_mappings = Vec::with_capacity(required_columns.len());
        for column in required_columns {
            let slot = resolve_output_slot(node.output_schema(), column)?;
            let field = node.output_schema().field(slot);
            key_mappings.push(SemijoinReductionKeyMappingDiagnostics {
                before_column: column_label(column),
                after_column: Some(field.name().clone()),
                target_subtree_id: Some(node_id),
            });
        }
        Ok(SemijoinLineageAnalysis {
            target_subtree_id: node_id,
            target_key_columns: required_columns.to_vec(),
            key_mappings,
            steps: Vec::new(),
        })
    }
}

struct SemijoinReductionRewriter<'a> {
    original: &'a QueryPlan,
    options: &'a SemijoinReductionOptions,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    diagnostics: SemijoinReductionDiagnostics,
    changed: bool,
}

impl<'a> SemijoinReductionRewriter<'a> {
    fn new(original: &'a QueryPlan, options: &'a SemijoinReductionOptions) -> Self {
        Self {
            original,
            options,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            diagnostics: SemijoinReductionDiagnostics::default(),
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        if let PlanNode::Join(join) = &old_node.kind {
            if join.spec.join_type == JoinType::LeftSemi {
                if let Some(new_id) = self.try_rewrite_left_semi_candidate(old_id, join) {
                    self.memo.insert(old_id, new_id);
                    return new_id;
                }
            }
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

    fn try_rewrite_left_semi_candidate(&mut self, candidate_node_id: NodeId, join: &JoinNode) -> Option<NodeId> {
        let mut candidate = SemijoinReductionCandidateDiagnostics {
            rule_name: SEMIJOIN_REDUCTION_RULE_NAME,
            candidate_node_id,
            before_root_node_id: candidate_node_id,
            after_root_node_id: None,
            crossed_join_node_ids: Vec::new(),
            target_subtree_id: None,
            key_mappings: Vec::new(),
            cost_gate: None,
            admission_decision: Some(SemijoinReductionAdmissionDecision::NotEvaluated),
            original_admitted: None,
            rewritten_admitted: None,
            applied: false,
            rejected_reason: None,
        };
        self.diagnostics.considered_count += 1;

        match self.plan_rewrite_for_candidate(join) {
            Ok(planned) => {
                candidate.after_root_node_id = Some(planned.new_id);
                candidate.crossed_join_node_ids =
                    planned.analysis.steps.iter().map(|step| step.crossed_join_node_id).collect();
                candidate.target_subtree_id = Some(planned.analysis.target_subtree_id);
                candidate.key_mappings = planned.analysis.key_mappings;
                candidate.cost_gate = Some(planned.cost_gate);
                candidate.applied = true;
                self.diagnostics.applied_count += 1;
                self.diagnostics.candidates.push(candidate);
                self.changed = true;
                Some(planned.new_id)
            },
            Err(rejection) => {
                if let Some(analysis) = rejection.analysis {
                    candidate.crossed_join_node_ids =
                        analysis.steps.iter().map(|step| step.crossed_join_node_id).collect();
                    candidate.target_subtree_id = Some(analysis.target_subtree_id);
                    candidate.key_mappings = analysis.key_mappings;
                }
                candidate.cost_gate = rejection.cost_gate;
                candidate.rejected_reason = Some(rejection.reason);
                self.diagnostics.candidates.push(candidate);
                None
            },
        }
    }

    fn plan_rewrite_for_candidate(&mut self, join: &JoinNode) -> Result<SemijoinRewritePlan, SemijoinRewriteRejection> {
        if join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left) {
            return Err(SemijoinRewriteRejection::new(SemijoinReductionRejectedReason::NonLeftOnlySemiProjection));
        }
        if join.spec.null_equality != JoinNullEquality::NullsDoNotMatch {
            return Err(SemijoinRewriteRejection::new(SemijoinReductionRejectedReason::UnsupportedNullEquality));
        }

        let analysis = match SemijoinLineageAnalyzer::new(self.original)
            .analyze_left_semi_key_lineage(join.inputs[0], &join.spec.keys)
        {
            Ok(analysis) => analysis,
            Err(reason) => return Err(SemijoinRewriteRejection::new(reason)),
        };

        if analysis.steps.is_empty() {
            return Err(SemijoinRewriteRejection::with_analysis(
                SemijoinReductionRejectedReason::UnsupportedLeftShape,
                analysis,
            ));
        }
        if analysis.steps.iter().any(|step| {
            crossed_join_null_equality(self.original, step.crossed_join_node_id)
                != Some(JoinNullEquality::NullsDoNotMatch)
        }) {
            return Err(SemijoinRewriteRejection::with_analysis(
                SemijoinReductionRejectedReason::UnsupportedNullEquality,
                analysis,
            ));
        }

        let cost_gate = evaluate_semijoin_cost_gate(
            self.original,
            join.inputs[1],
            analysis.target_subtree_id,
            self.options.cost_threshold_multiplier,
        );
        if !cost_gate.accepted {
            return Err(SemijoinRewriteRejection {
                reason: SemijoinReductionRejectedReason::CostGateRejected,
                analysis: Some(analysis),
                cost_gate: Some(cost_gate),
            });
        }

        let pushed_left = self.build_pushed_subtree(join.inputs[0], join, &analysis, 0);
        let new_id = self.apply_original_semijoin_projection_if_needed(join, pushed_left);
        Ok(SemijoinRewritePlan { new_id, analysis, cost_gate })
    }

    fn build_pushed_subtree(
        &mut self,
        current_old_id: NodeId,
        original_semijoin: &JoinNode,
        analysis: &SemijoinLineageAnalysis,
        step_index: usize,
    ) -> NodeId {
        if step_index == analysis.steps.len() {
            return self.build_pushed_semijoin(current_old_id, original_semijoin, analysis);
        }

        let step = &analysis.steps[step_index];
        debug_assert_eq!(step.crossed_join_node_id, current_old_id);
        let node = self.original.node(current_old_id).expect("crossed join node must exist");
        let PlanNode::Join(join) = &node.kind else {
            panic!("lineage step must point at a join node");
        };

        let left = if step.target_side == JoinSide::Left {
            self.build_pushed_subtree(step.target_child_id, original_semijoin, analysis, step_index + 1)
        } else {
            self.rewrite_subtree(join.inputs[0])
        };
        let right = if step.target_side == JoinSide::Right {
            self.build_pushed_subtree(step.target_child_id, original_semijoin, analysis, step_index + 1)
        } else {
            self.rewrite_subtree(join.inputs[1])
        };

        let rebuilt = JoinNode { inputs: [left, right], spec: join.spec.clone() };
        self.push_node(PlanNode::Join(rebuilt), node.facts.clone())
    }

    fn build_pushed_semijoin(
        &mut self,
        target_old_id: NodeId,
        original_semijoin: &JoinNode,
        analysis: &SemijoinLineageAnalysis,
    ) -> NodeId {
        let target_new_id = self.rewrite_subtree(target_old_id);
        let right_new_id = self.rewrite_subtree(original_semijoin.inputs[1]);
        let keys = original_semijoin
            .spec
            .keys
            .iter()
            .zip(&analysis.target_key_columns)
            .map(|(key, target_column)| {
                JoinKey::new(input_column(JoinSide::Left, target_column), input_column(JoinSide::Right, &key.right))
            })
            .collect::<Vec<_>>();

        let target_schema =
            self.nodes.get(target_new_id.index()).expect("rewritten target node must exist").output_schema().clone();
        let mut spec = original_semijoin.spec.clone();
        spec.keys = keys;
        spec.output_projection = left_output_projection_for_schema(&target_schema);
        spec.build_artifact_id = None;

        self.push_node(
            PlanNode::Join(JoinNode { inputs: [target_new_id, right_new_id], spec }),
            PlanFacts::device(target_schema, SchemaBehavior::Join),
        )
    }

    fn apply_original_semijoin_projection_if_needed(
        &mut self,
        original_semijoin: &JoinNode,
        pushed_left: NodeId,
    ) -> NodeId {
        let pushed_schema = self.nodes[pushed_left.index()].output_schema().clone();
        let projected_schema = semijoin_projection_schema(self.original, original_semijoin);
        if schemas_equal(&pushed_schema, &projected_schema)
            && semijoin_projection_is_identity(self.original, original_semijoin)
        {
            return pushed_left;
        }

        let outputs = original_semijoin
            .spec
            .output_projection
            .iter()
            .zip(projected_schema.fields().iter())
            .map(|(column, field)| {
                ProjectionOutput::new(
                    Expr::column(column.column.name()).alias(column.output_name.clone()),
                    field.as_ref().clone(),
                )
            })
            .collect::<Vec<_>>();
        self.push_node(
            PlanNode::Projection(crate::plan::ProjectionNode::new(pushed_left, outputs)),
            PlanFacts::device(projected_schema, SchemaBehavior::Projection),
        )
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

fn prepend_mappings(
    existing: &mut [SemijoinReductionKeyMappingDiagnostics],
    local_mappings: Vec<SemijoinReductionKeyMappingDiagnostics>,
) {
    for (mapping, local) in existing.iter_mut().zip(local_mappings) {
        mapping.before_column = local.before_column;
    }
}

fn simple_projection_column(expr: &Expr) -> Option<&ColumnRef> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Alias { expr, .. } => match expr.as_ref() {
            Expr::Column(column) => Some(column),
            _ => None,
        },
        _ => None,
    }
}

fn resolve_output_slot(
    schema: &arrow::datatypes::SchemaRef,
    column: &ColumnRef,
) -> Result<usize, SemijoinReductionRejectedReason> {
    let mut matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| (field.name() == column.name()).then_some(index));
    let Some(index) = matches.next() else {
        return Err(SemijoinReductionRejectedReason::DroppedKeyLineage);
    };
    if matches.next().is_some() {
        return Err(SemijoinReductionRejectedReason::AmbiguousKeyLineage);
    }
    Ok(index)
}

fn validate_join_output_column_qualifier(
    side: JoinSide,
    column: &ColumnRef,
) -> Result<(), SemijoinReductionRejectedReason> {
    let expected_input = join_side_index(side);
    match column.qualifier() {
        Some(ColumnQualifier::Input { index }) if *index == expected_input => Ok(()),
        _ => Err(SemijoinReductionRejectedReason::AmbiguousKeyLineage),
    }
}

fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

fn column_label(column: &ColumnRef) -> String {
    column.name().to_string()
}

fn crossed_join_null_equality(plan: &QueryPlan, node_id: NodeId) -> Option<JoinNullEquality> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Join(join) => Some(join.spec.null_equality),
        _ => None,
    }
}

fn input_column(side: JoinSide, column: &ColumnRef) -> ColumnRef {
    ColumnRef::input(join_side_index(side), column.name(), column.display_name())
}

fn evaluate_semijoin_cost_gate(
    plan: &QueryPlan,
    semi_right: NodeId,
    target_subtree: NodeId,
    threshold_multiplier: u64,
) -> SemijoinReductionCostGateDiagnostics {
    let threshold_multiplier = threshold_multiplier.max(1);
    if let (Some(semi_right_rows), Some(target_rows)) =
        (estimated_output_rows(plan, semi_right), estimated_output_rows(plan, target_subtree))
    {
        return SemijoinReductionCostGateDiagnostics {
            signal: SemijoinReductionCostSignal::RowEstimate { semi_right_rows, target_rows },
            threshold_multiplier,
            accepted: semi_right_rows.saturating_mul(threshold_multiplier) <= target_rows,
        };
    }

    if let (Some(semi_right_bytes), Some(target_bytes)) =
        (estimated_output_bytes(plan, semi_right), estimated_output_bytes(plan, target_subtree))
    {
        return SemijoinReductionCostGateDiagnostics {
            signal: SemijoinReductionCostSignal::ByteEstimate { semi_right_bytes, target_bytes },
            threshold_multiplier,
            accepted: semi_right_bytes.saturating_mul(threshold_multiplier) <= target_bytes,
        };
    }

    let accepted = is_aggregate_keyset_shape(plan, semi_right);
    SemijoinReductionCostGateDiagnostics {
        signal: if accepted {
            SemijoinReductionCostSignal::AggregateKeysetShape
        } else {
            SemijoinReductionCostSignal::Unavailable
        },
        threshold_multiplier,
        accepted,
    }
}

fn is_aggregate_keyset_shape(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Filter(filter) => is_aggregate_keyset_shape(plan, filter.input),
        PlanNode::Projection(projection) => is_aggregate_keyset_shape(plan, projection.input),
        PlanNode::Aggregate(aggregate) => {
            !aggregate.spec.group_exprs.is_empty() && is_source_derived_keyset_input(plan, aggregate.input)
        },
        PlanNode::Source(_) | PlanNode::Join(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => false,
    }
}

fn is_source_derived_keyset_input(plan: &QueryPlan, node_id: NodeId) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Source(_) => true,
        PlanNode::Filter(filter) => is_source_derived_keyset_input(plan, filter.input),
        PlanNode::Projection(projection) => is_source_derived_keyset_input(plan, projection.input),
        PlanNode::Aggregate(_) | PlanNode::Join(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => false,
    }
}

fn left_output_projection_for_schema(schema: &arrow::datatypes::SchemaRef) -> Vec<JoinOutputColumn> {
    schema
        .fields()
        .iter()
        .map(|field| {
            JoinOutputColumn::left(
                ColumnRef::input(0, field.name().as_str(), field.name().as_str()),
                field.name().clone(),
            )
        })
        .collect()
}

fn semijoin_projection_schema(plan: &QueryPlan, join: &JoinNode) -> arrow::datatypes::SchemaRef {
    let left_schema = plan.node(join.inputs[0]).expect("left-semi left input must exist").output_schema().clone();
    let fields = join
        .spec
        .output_projection
        .iter()
        .map(|output| {
            let slot = resolve_output_slot(&left_schema, &output.column)
                .expect("left-semi projection should resolve before rewrite");
            let input_field = left_schema.field(slot);
            Field::new(output.output_name.clone(), input_field.data_type().clone(), input_field.is_nullable())
        })
        .collect::<Vec<_>>();
    std::sync::Arc::new(Schema::new(fields))
}

fn semijoin_projection_is_identity(plan: &QueryPlan, join: &JoinNode) -> bool {
    let left_schema = plan.node(join.inputs[0]).expect("left-semi left input must exist").output_schema();
    if join.spec.output_projection.len() != left_schema.fields().len() {
        return false;
    }

    join.spec.output_projection.iter().zip(left_schema.fields().iter()).all(|(output, field)| {
        output.side == JoinSide::Left && output.column.name() == field.name() && output.output_name == *field.name()
    })
}

fn schemas_equal(left: &arrow::datatypes::SchemaRef, right: &arrow::datatypes::SchemaRef) -> bool {
    left.as_ref() == right.as_ref()
}

fn validate_rewritten_root_semantics(
    original: &QueryPlan,
    rewritten: &QueryPlan,
) -> Result<(), SemijoinReductionRejectedReason> {
    if !schemas_equal(original.output_schema(), rewritten.output_schema()) {
        return Err(SemijoinReductionRejectedReason::RootSchemaMismatch);
    }
    if !schemas_equal(original.root_node().output_schema(), rewritten.root_node().output_schema()) {
        return Err(SemijoinReductionRejectedReason::RootOutputProjectionMismatch);
    }
    Ok(())
}

fn evaluate_admission_gate(
    original: &QueryPlan,
    rewritten: &QueryPlan,
    mode: SemijoinReductionAdmissionMode,
) -> AdmissionGateDecision {
    match mode {
        SemijoinReductionAdmissionMode::NotEvaluated => AdmissionGateDecision::Keep {
            decision: SemijoinReductionAdmissionDecision::NotEvaluated,
            original_admitted: None,
            rewritten_admitted: None,
        },
        SemijoinReductionAdmissionMode::StrictBounded => mode_admission_gate_decision(
            strict_bounded_admitted(original),
            strict_bounded_admitted(rewritten),
            SemijoinReductionAdmissionDecision::StrictBoundedKept,
            SemijoinReductionAdmissionDecision::StrictBoundedRejectedRewrite,
        ),
        SemijoinReductionAdmissionMode::Adaptive => mode_admission_gate_decision(
            adaptive_admitted(original),
            adaptive_admitted(rewritten),
            SemijoinReductionAdmissionDecision::AdaptiveKept,
            SemijoinReductionAdmissionDecision::AdaptiveRejectedRewrite,
        ),
    }
}

fn mode_admission_gate_decision(
    original_admitted: Option<bool>,
    rewritten_admitted: Option<bool>,
    kept: SemijoinReductionAdmissionDecision,
    rejected: SemijoinReductionAdmissionDecision,
) -> AdmissionGateDecision {
    let keep = matches!((original_admitted, rewritten_admitted), (Some(true), Some(true)) | (Some(false), Some(true)));
    if keep {
        AdmissionGateDecision::Keep { decision: kept, original_admitted, rewritten_admitted }
    } else {
        AdmissionGateDecision::Reject { decision: rejected, original_admitted, rewritten_admitted }
    }
}

fn strict_bounded_admitted(plan: &QueryPlan) -> Option<bool> {
    match crate::exec::strict_bounded_pipeline_admission(plan) {
        Ok(Some(_)) => Some(true),
        Ok(None) => Some(false),
        Err(error) if error.code() == crate::error::QueryErrorCode::UnsupportedPlanShape => Some(false),
        Err(_) => None,
    }
}

fn adaptive_admitted(plan: &QueryPlan) -> Option<bool> {
    match crate::exec::adaptive_bounded_pipeline_admission(plan) {
        Ok(Some(admission)) => Some(admission.report().row().candidate_status == AdaptiveCandidateStatus::Admitted),
        Ok(None) => Some(false),
        Err(_) => None,
    }
}

fn record_admission_for_applied_candidates(
    diagnostics: &mut SemijoinReductionDiagnostics,
    decision: SemijoinReductionAdmissionDecision,
    original_admitted: Option<bool>,
    rewritten_admitted: Option<bool>,
) {
    for candidate in diagnostics.candidates.iter_mut().filter(|candidate| candidate.applied) {
        candidate.admission_decision = Some(decision.clone());
        candidate.original_admitted = original_admitted;
        candidate.rewritten_admitted = rewritten_admitted;
    }
}

fn reject_applied_candidates(
    diagnostics: &mut SemijoinReductionDiagnostics,
    reason: SemijoinReductionRejectedReason,
    decision: Option<SemijoinReductionAdmissionDecision>,
    original_admitted: Option<bool>,
    rewritten_admitted: Option<bool>,
) {
    for candidate in diagnostics.candidates.iter_mut().filter(|candidate| candidate.applied) {
        candidate.applied = false;
        candidate.rejected_reason = Some(reason);
        if let Some(decision) = decision.clone() {
            candidate.admission_decision = Some(decision);
        }
        candidate.original_admitted = original_admitted;
        candidate.rewritten_admitted = rewritten_admitted;
    }
    diagnostics.applied_count = diagnostics.candidates.iter().filter(|candidate| candidate.applied).count();
}

fn sink_with_input(sink: &ResultSink, input: NodeId) -> ResultSink {
    match sink {
        ResultSink::GpuResident { .. } => ResultSink::GpuResident { input },
        ResultSink::HostArrow { schema, .. } => ResultSink::HostArrow { input, schema: schema.clone() },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateSpec, BinaryOp, LimitSpec, SortExpr, SortSpec},
        plan::{JoinOutputColumn, JoinSpec, ProjectionOutput, QueryPlanBuilder},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    #[test]
    fn lineage_accepts_source_passthrough_key() {
        let (plan, source) = single_source_plan(vec![field("k"), field("payload")]);
        let analysis = analyze_keys(&plan, source, &[left_key("k")]).expect("source key should map");

        assert_eq!(analysis.target_subtree_id, source);
        assert!(analysis.steps.is_empty());
        assert_eq!(analysis.key_mappings[0].after_column.as_deref(), Some("k"));
    }

    #[test]
    fn lineage_accepts_filter_and_simple_alias_projection() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("orders", vec![field("o_orderkey"), field("payload")]));
        let filter =
            builder.add_filter(source, Expr::column("payload"), schema(vec![field("o_orderkey"), field("payload")]));
        let projection = builder.add_projection(
            filter,
            vec![ProjectionOutput::new(
                Expr::column("o_orderkey").alias("orderkey"),
                Field::new("orderkey", DataType::Int32, false),
            )],
        );
        let plan = builder.finish_gpu_resident(projection);

        let analysis =
            analyze_keys(&plan, projection, &[left_key("orderkey")]).expect("alias should map to source key");

        assert_eq!(analysis.target_subtree_id, source);
        assert_eq!(analysis.key_mappings[0].before_column, "orderkey");
        assert_eq!(analysis.key_mappings[0].after_column.as_deref(), Some("o_orderkey"));
    }

    #[test]
    fn lineage_records_join_output_side_for_left_child_key() {
        let (plan, join, left, _) = two_source_join_plan(
            JoinType::Inner,
            vec![
                JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "join_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "r_payload", "r_payload"), "r_payload"),
            ],
        );

        let analysis = analyze_keys(&plan, join, &[left_key("join_key")]).expect("join key should map to left child");

        assert_eq!(analysis.target_subtree_id, left);
        assert_eq!(analysis.steps.len(), 1);
        assert_eq!(analysis.steps[0].crossed_join_node_id, join);
        assert_eq!(analysis.steps[0].target_side, JoinSide::Left);
        assert_eq!(analysis.steps[0].target_child_id, left);
    }

    #[test]
    fn lineage_records_join_output_side_for_right_child_key() {
        let (plan, join, _, right) = two_source_join_plan(
            JoinType::Inner,
            vec![
                JoinOutputColumn::left(ColumnRef::input(0, "l_payload", "l_payload"), "l_payload"),
                JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "join_key"),
            ],
        );

        let analysis = analyze_keys(&plan, join, &[left_key("join_key")]).expect("join key should map to right child");

        assert_eq!(analysis.target_subtree_id, right);
        assert_eq!(analysis.steps[0].target_side, JoinSide::Right);
    }

    #[test]
    fn lineage_reaches_deepest_supported_inner_join_subtree() {
        let mut builder = QueryPlanBuilder::new();
        let a = builder.add_source(source("a", vec![field("a_key")]));
        let b = builder.add_source(source("b", vec![field("b_key")]));
        let c = builder.add_source(source("c", vec![field("c_key")]));
        let ab = builder.add_join(
            a,
            b,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "a_key", "a_key"), ColumnRef::input(1, "b_key", "b_key"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "b_key", "b_key"), "b_key"),
            ]),
            schema(vec![field("a_key"), field("b_key")]),
        );
        let abc = builder.add_join(
            ab,
            c,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "c_key", "c_key"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "b_key", "b_key"), "join_key")]),
            schema(vec![Field::new("join_key", DataType::Int32, false)]),
        );
        let plan = builder.finish_gpu_resident(abc);

        let analysis = analyze_keys(&plan, abc, &[left_key("join_key")]).expect("nested key should map to b");

        assert_eq!(analysis.target_subtree_id, b);
        assert_eq!(analysis.steps.len(), 2);
        assert_eq!(analysis.steps[0].crossed_join_node_id, abc);
        assert_eq!(analysis.steps[0].target_side, JoinSide::Left);
        assert_eq!(analysis.steps[1].crossed_join_node_id, ab);
        assert_eq!(analysis.steps[1].target_side, JoinSide::Right);
    }

    #[test]
    fn lineage_rejects_expression_derived_projection_key() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("orders", vec![field("k")]));
        let projection = builder.add_projection(
            source,
            vec![ProjectionOutput::new(
                Expr::binary(Expr::column("k"), BinaryOp::Add, Expr::column("k")),
                Field::new("k2", DataType::Int32, false),
            )],
        );
        let plan = builder.finish_gpu_resident(projection);

        assert_eq!(
            analyze_keys(&plan, projection, &[left_key("k2")]).unwrap_err(),
            SemijoinReductionRejectedReason::ExpressionDerivedKey
        );
    }

    #[test]
    fn lineage_rejects_ambiguous_duplicate_output_names() {
        let (plan, source) = single_source_plan(vec![field("k"), field("k")]);

        assert_eq!(
            analyze_keys(&plan, source, &[left_key("k")]).unwrap_err(),
            SemijoinReductionRejectedReason::AmbiguousKeyLineage
        );
    }

    #[test]
    fn lineage_rejects_dropped_join_projection_key() {
        let (plan, join, _, _) = two_source_join_plan(
            JoinType::Inner,
            vec![JoinOutputColumn::right(ColumnRef::input(1, "r_payload", "r_payload"), "r_payload")],
        );

        assert_eq!(
            analyze_keys(&plan, join, &[left_key("l_key")]).unwrap_err(),
            SemijoinReductionRejectedReason::DroppedKeyLineage
        );
    }

    #[test]
    fn lineage_rejects_required_keys_that_map_to_both_join_sides() {
        let (plan, join, _, _) = two_source_join_plan(
            JoinType::Inner,
            vec![
                JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
            ],
        );

        assert_eq!(
            analyze_keys(&plan, join, &[left_key("l_key"), left_key("r_key")]).unwrap_err(),
            SemijoinReductionRejectedReason::AmbiguousKeyLineage
        );
    }

    #[test]
    fn lineage_rejects_non_inner_crossed_join() {
        let (plan, join, _, _) = two_source_join_plan(
            JoinType::Left,
            vec![JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key")],
        );

        assert_eq!(
            analyze_keys(&plan, join, &[left_key("l_key")]).unwrap_err(),
            SemijoinReductionRejectedReason::UnsupportedCrossedJoinType
        );
    }

    #[test]
    fn lineage_rejects_aggregate_sort_and_limit_boundaries() {
        for plan_case in [unsupported_aggregate_plan(), unsupported_sort_plan(), unsupported_limit_plan()] {
            assert_eq!(
                analyze_keys(&plan_case.0, plan_case.1, &[left_key("k")]).unwrap_err(),
                SemijoinReductionRejectedReason::UnsupportedLeftShape
            );
        }
    }

    #[test]
    fn cost_gate_accepts_selective_row_estimate() {
        let mut builder = QueryPlanBuilder::new();
        let target = builder.add_source(local_source("target", 1_000, Some(8_000)));
        let semi_right = builder.add_source(local_source("semi_right", 100, Some(800)));
        let plan = builder.finish_gpu_resident(target);

        let gate = evaluate_semijoin_cost_gate(&plan, semi_right, target, 8);

        assert!(gate.accepted);
        assert_eq!(gate.threshold_multiplier, 8);
        assert_eq!(gate.signal, SemijoinReductionCostSignal::RowEstimate { semi_right_rows: 100, target_rows: 1_000 });
    }

    #[test]
    fn cost_gate_rejects_unselective_row_estimate() {
        let mut builder = QueryPlanBuilder::new();
        let target = builder.add_source(local_source("target", 700, Some(7_000)));
        let semi_right = builder.add_source(local_source("semi_right", 100, Some(800)));
        let plan = builder.finish_gpu_resident(target);

        let gate = evaluate_semijoin_cost_gate(&plan, semi_right, target, 8);

        assert!(!gate.accepted);
        assert_eq!(gate.signal, SemijoinReductionCostSignal::RowEstimate { semi_right_rows: 100, target_rows: 700 });
    }

    #[test]
    fn cost_gate_falls_back_to_byte_estimate_when_rows_are_unavailable() {
        let mut builder = QueryPlanBuilder::new();
        let target = builder.add_source(local_source_with_bytes_only("target", 8_000));
        let semi_right = builder.add_source(local_source_with_bytes_only("semi_right", 1_100));
        let plan = builder.finish_gpu_resident(target);

        let gate = evaluate_semijoin_cost_gate(&plan, semi_right, target, 8);

        assert!(!gate.accepted);
        assert_eq!(
            gate.signal,
            SemijoinReductionCostSignal::ByteEstimate { semi_right_bytes: 1_100, target_bytes: 8_000 }
        );
    }

    fn analyze_keys(
        plan: &QueryPlan,
        left_input: NodeId,
        keys: &[JoinKey],
    ) -> Result<SemijoinLineageAnalysis, SemijoinReductionRejectedReason> {
        SemijoinLineageAnalyzer::new(plan).analyze_left_semi_key_lineage(left_input, keys)
    }

    fn single_source_plan(fields: Vec<Field>) -> (QueryPlan, NodeId) {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", fields));
        let plan = builder.finish_gpu_resident(source);
        (plan, source)
    }

    fn two_source_join_plan(
        join_type: JoinType,
        output_projection: Vec<JoinOutputColumn>,
    ) -> (QueryPlan, NodeId, NodeId, NodeId) {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(source("left", vec![field("l_key"), field("l_payload")]));
        let right = builder.add_source(source("right", vec![field("r_key"), field("r_payload")]));
        let output_schema = Arc::new(Schema::new(
            output_projection
                .iter()
                .map(|output| Field::new(output.output_name.as_str(), DataType::Int32, false))
                .collect::<Vec<_>>(),
        ));
        let join = builder.add_join(
            left,
            right,
            JoinSpec::new(
                join_type,
                vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
            )
            .with_output_projection(output_projection),
            output_schema,
        );
        let plan = builder.finish_gpu_resident(join);
        (plan, join, left, right)
    }

    fn unsupported_aggregate_plan() -> (QueryPlan, NodeId) {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", vec![field("k")]));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(vec![Expr::column("k")], vec![]),
            schema(vec![field("k")]),
        );
        let plan = builder.finish_gpu_resident(aggregate);
        (plan, aggregate)
    }

    fn unsupported_sort_plan() -> (QueryPlan, NodeId) {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", vec![field("k")]));
        let sort = builder.add_sort(
            source,
            SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]),
            schema(vec![field("k")]),
        );
        let plan = builder.finish_gpu_resident(sort);
        (plan, sort)
    }

    fn unsupported_limit_plan() -> (QueryPlan, NodeId) {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", vec![field("k")]));
        let limit = builder.add_limit(source, LimitSpec::fetch(10), schema(vec![field("k")]));
        let plan = builder.finish_gpu_resident(limit);
        (plan, limit)
    }

    fn source(name: &str, fields: Vec<Field>) -> SourceNode {
        SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
    }

    fn local_source(name: &str, rows: u64, bytes: Option<u64>) -> SourceNode {
        let file = LocalParquetFile::new(format!("/tmp/{name}.parquet")).with_row_group_rows(vec![rows]);
        let file =
            if let Some(bytes) = bytes { file.with_projected_row_group_bytes(vec![bytes], vec![bytes]) } else { file };
        SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![file],
            schema(vec![field("k")]),
            vec!["k".to_string()],
        ))
    }

    fn local_source_with_bytes_only(name: &str, bytes: u64) -> SourceNode {
        SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(format!("/tmp/{name}.parquet"))
                    .with_projected_row_group_bytes(vec![bytes], vec![bytes]),
            ],
            schema(vec![field("k")]),
            vec!["k".to_string()],
        ))
    }

    fn schema(fields: Vec<Field>) -> Arc<Schema> {
        Arc::new(Schema::new(fields))
    }

    fn field(name: &str) -> Field {
        Field::new(name, DataType::Int32, false)
    }

    fn left_key(name: &str) -> JoinKey {
        JoinKey::new(ColumnRef::input(0, name, name), ColumnRef::input(1, name, name))
    }
}
