use std::collections::BTreeMap;

use arrow::datatypes::{Field, Schema, SchemaRef as ArrowSchemaRef};

use super::{rewrite, simple_projection_column_single_alias, source_target_policy::OptimizerSourceTargetPolicy};
use crate::{
    adaptive_report::AdaptiveCandidateStatus,
    error::{QueryError, QueryErrorCode},
    expr::{ColumnQualifier, ColumnRef, Expr},
    plan::{
        JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinType, NodeId, PlanFacts, PlanNode,
        ProjectionOutput, QueryNode, QueryPlan, SchemaBehavior, estimated_output_bytes, estimated_output_rows,
    },
};

/// Stable rule name used in optimizer diagnostics.
pub const SEMIJOIN_REDUCTION_RULE_NAME: &str = "semijoin_reduction";
/// Default cost gate multiplier for comparing semijoin-side cost to target cost.
pub const DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER: u64 =
    SemijoinReductionOptions::DEFAULT_COST_THRESHOLD_MULTIPLIER;

/// Configuration for the semijoin-reduction optimizer pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionOptions {
    /// Whether the pass is allowed to rewrite eligible left-semi joins.
    pub enabled: bool,
    /// Maximum accepted `semi_side / target_side` cost multiplier.
    pub cost_threshold_multiplier: u64,
    /// Optional post-rewrite admission gate used to protect execution modes.
    ///
    /// The gate rejects a rewrite only when the original plan is already
    /// admitted by the selected mode and the rewritten plan is not. If neither
    /// shape is admitted yet, the rewrite is kept so later native optimizer
    /// passes can still turn the semantically-reduced plan into an admitted
    /// execution shape. Unknown admission errors remain conservative and reject
    /// the tentative rewrite.
    pub admission_mode: SemijoinReductionAdmissionMode,
}

impl SemijoinReductionOptions {
    /// Default cost gate multiplier for comparing semijoin-side cost to target cost.
    pub const DEFAULT_COST_THRESHOLD_MULTIPLIER: u64 = 8;

    /// Returns options with the pass disabled and default thresholds preserved.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            cost_threshold_multiplier: Self::DEFAULT_COST_THRESHOLD_MULTIPLIER,
            admission_mode: SemijoinReductionAdmissionMode::NotEvaluated,
        }
    }

    /// Sets the mode-specific admission gate used after tentative rewrites.
    pub fn with_admission_mode(mut self, admission_mode: SemijoinReductionAdmissionMode) -> Self {
        self.admission_mode = admission_mode;
        self
    }

    /// Validates internal consistency for semijoin-reduction cost gates.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.cost_threshold_multiplier == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.semijoin_reduction_options.validate",
                "semijoin-reduction option must be greater than zero",
            )
            .with_fact("field", "cost_threshold_multiplier")
            .with_fact("value", self.cost_threshold_multiplier)
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for SemijoinReductionOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            cost_threshold_multiplier: Self::DEFAULT_COST_THRESHOLD_MULTIPLIER,
            admission_mode: SemijoinReductionAdmissionMode::NotEvaluated,
        }
    }
}

/// Mode-specific admission gate to use after a tentative semijoin rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemijoinReductionAdmissionMode {
    /// Do not run a post-rewrite admission gate.
    NotEvaluated,
    /// Reject only when strict bounded admission would regress an already
    /// admitted original plan.
    StrictBounded,
    /// Reject only when adaptive admission would regress an already admitted
    /// original plan.
    Adaptive,
}

/// Result for one semijoin-reduction optimizer pass invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct SemijoinReductionResult {
    /// Plan returned after the pass, rewritten only when the pass was kept.
    pub plan: QueryPlan,
    /// Whether this pass changed the plan.
    pub change: super::NativeOptimizerChange,
    /// Candidate diagnostics collected by the pass.
    pub diagnostics: SemijoinReductionDiagnostics,
}

/// Diagnostics for semijoin-reduction candidates considered by the optimizer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SemijoinReductionDiagnostics {
    /// Number of left-semi candidates considered.
    pub considered_count: usize,
    /// Number of candidates applied and kept.
    pub applied_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<SemijoinReductionCandidateDiagnostics>,
}

impl SemijoinReductionDiagnostics {
    /// Returns true when the pass produced no candidate evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    /// Merges another semijoin diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate semijoin-reduction evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Left-semi join candidate node id before rewrite.
    pub candidate_node_id: NodeId,
    /// Root node id before the tentative rewrite.
    pub before_root_node_id: NodeId,
    /// Root node id after rewrite when a tentative rewrite was built.
    pub after_root_node_id: Option<NodeId>,
    /// Join node ids crossed while pushing the semijoin reduction.
    pub crossed_join_node_ids: Vec<NodeId>,
    /// Target subtree that would receive the pushed semijoin filter.
    pub target_subtree_id: Option<NodeId>,
    /// Key lineage mappings from the original semijoin to the target subtree.
    pub key_mappings: Vec<SemijoinReductionKeyMappingDiagnostics>,
    /// Cost-gate evidence when lineage reached a target subtree.
    pub cost_gate: Option<SemijoinReductionCostGateDiagnostics>,
    /// Configured cost multiplier used by the semijoin cost gate.
    pub cost_threshold_multiplier: u64,
    /// Configured post-rewrite admission gate mode.
    pub admission_mode: SemijoinReductionAdmissionMode,
    /// Admission decision used after tentative rewrite, if evaluated.
    pub admission_decision: Option<SemijoinReductionAdmissionDecision>,
    /// Whether the original plan was admitted by the selected admission mode.
    pub original_admitted: Option<bool>,
    /// Whether the rewritten plan was admitted by the selected admission mode.
    pub rewritten_admitted: Option<bool>,
    /// True when this candidate's rewrite was applied to the returned plan.
    pub applied: bool,
    /// Rejection reason when the candidate was not applied.
    pub rejected_reason: Option<SemijoinReductionRejectedReason>,
}

/// One left-semi key mapping before and after a pushdown step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionKeyMappingDiagnostics {
    /// Column name referenced by the original semijoin key.
    pub before_column: String,
    /// Column name after crossing projections/joins, if lineage survived.
    pub after_column: Option<String>,
    /// Target subtree reached by this mapping.
    pub target_subtree_id: Option<NodeId>,
}

/// Cost-gate evidence used to decide whether a semijoin pushdown is worth keeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionCostGateDiagnostics {
    /// Cost signal used for the keep/reject decision.
    pub signal: SemijoinReductionCostSignal,
    /// Threshold multiplier configured for this pass.
    pub threshold_multiplier: u64,
    /// Whether the cost gate accepted the candidate.
    pub accepted: bool,
}

/// Cost signal used by semijoin reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemijoinReductionCostSignal {
    /// Row-count estimates were available for both compared sides.
    RowEstimate {
        /// Estimated rows on the semijoin right side.
        semi_right_rows: u64,
        /// Estimated rows for the target subtree.
        target_rows: u64,
    },
    /// Byte estimates were available for both compared sides.
    ByteEstimate {
        /// Estimated bytes on the semijoin right side.
        semi_right_bytes: u64,
        /// Estimated bytes for the target subtree.
        target_bytes: u64,
    },
    /// Aggregate keyset shape was considered small enough without estimates.
    AggregateKeysetShape,
    /// A filtered source-derived keyset was pushed into a dimension-sized
    /// subtree so downstream joins can reuse the narrowed dimension keys.
    FilteredSourceKeysetShape {
        /// Number of static filter or reader-predicate nodes found in the
        /// semijoin right-side keyset subtree.
        filtered_source_count: usize,
        /// Estimated source bytes scanned by the semijoin right-side keyset
        /// subtree, when every source has byte evidence.
        semi_right_source_bytes: Option<u64>,
        /// Estimated rows for the target subtree that receives the pushed
        /// semijoin.
        target_rows: Option<u64>,
        /// Estimated bytes for the target subtree that receives the pushed
        /// semijoin.
        target_bytes: Option<u64>,
        /// Number of inner joins crossed while pushing the semijoin.
        crossed_join_count: usize,
    },
    /// No usable cost signal was available.
    Unavailable,
}

/// Mode-specific admission decision that caused a rewrite to be kept or rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemijoinReductionAdmissionDecision {
    /// No admission gate was evaluated.
    NotEvaluated,
    /// Strict bounded admission gate kept the rewritten plan.
    StrictBoundedKept,
    /// Strict bounded admission rejected the rewritten plan.
    StrictBoundedRejectedRewrite,
    /// Adaptive admission gate kept the rewritten plan.
    AdaptiveKept,
    /// Adaptive admission rejected the rewritten plan.
    AdaptiveRejectedRewrite,
}

/// Generic semijoin-reduction rejection reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemijoinReductionRejectedReason {
    /// Pass was disabled.
    PassDisabled,
    /// Candidate node was not a left-semi join.
    NotLeftSemiJoin,
    /// Left-semi shape was not supported by this pass.
    UnsupportedLeftShape,
    /// A crossed join type was not safe for semijoin pushdown.
    UnsupportedCrossedJoinType,
    /// Join null-equality semantics were unsupported.
    UnsupportedNullEquality,
    /// Required key lineage could not be found.
    MissingKeyLineage,
    /// Key lineage matched more than one possible target.
    AmbiguousKeyLineage,
    /// Key lineage passed through a derived expression.
    ExpressionDerivedKey,
    /// Key lineage was dropped before reaching the target.
    DroppedKeyLineage,
    /// Semijoin projection referenced non-left-side output.
    NonLeftOnlySemiProjection,
    /// Cost gate rejected the candidate.
    CostGateRejected,
    /// Rewritten root schema did not match the original root schema.
    RootSchemaMismatch,
    /// Rewritten root output projection did not match the original.
    RootOutputProjectionMismatch,
    /// Mode-specific admission rejected the rewritten plan.
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

    let rewritten_plan = rewrite::finish_rewritten_plan(rewriter.nodes, new_root, plan.sink())
        .with_keyed_hash_join_strategy_options_from(&plan);
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

type SemijoinRewriteResult = Result<SemijoinRewritePlan, Box<SemijoinRewriteRejection>>;

struct SemijoinRewriteRejection {
    reason: SemijoinReductionRejectedReason,
    analysis: Option<SemijoinLineageAnalysis>,
    cost_gate: Option<SemijoinReductionCostGateDiagnostics>,
}

impl SemijoinRewriteRejection {
    fn new(reason: SemijoinReductionRejectedReason) -> Box<Self> {
        Box::new(Self { reason, analysis: None, cost_gate: None })
    }

    fn with_analysis(reason: SemijoinReductionRejectedReason, analysis: SemijoinLineageAnalysis) -> Box<Self> {
        Box::new(Self { reason, analysis: Some(analysis), cost_gate: None })
    }

    fn with_analysis_and_cost_gate(
        reason: SemijoinReductionRejectedReason,
        analysis: SemijoinLineageAnalysis,
        cost_gate: SemijoinReductionCostGateDiagnostics,
    ) -> Box<Self> {
        Box::new(Self { reason, analysis: Some(analysis), cost_gate: Some(cost_gate) })
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
                    let input_column = simple_projection_column_single_alias(&output.expr)
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
            PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => Err(SemijoinReductionRejectedReason::UnsupportedLeftShape),
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
        if let PlanNode::Join(join) = &old_node.kind
            && join.spec.join_type == JoinType::LeftSemi
            && let Some(new_id) = self.try_rewrite_left_semi_candidate(old_id, join)
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
            cost_threshold_multiplier: self.options.cost_threshold_multiplier,
            admission_mode: self.options.admission_mode,
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
                let SemijoinRewriteRejection { reason, analysis, cost_gate } = *rejection;
                if let Some(analysis) = analysis {
                    candidate.crossed_join_node_ids =
                        analysis.steps.iter().map(|step| step.crossed_join_node_id).collect();
                    candidate.target_subtree_id = Some(analysis.target_subtree_id);
                    candidate.key_mappings = analysis.key_mappings;
                }
                candidate.cost_gate = cost_gate;
                candidate.rejected_reason = Some(reason);
                self.diagnostics.candidates.push(candidate);
                None
            },
        }
    }

    fn plan_rewrite_for_candidate(&mut self, join: &JoinNode) -> SemijoinRewriteResult {
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
            &analysis,
            self.options.cost_threshold_multiplier,
        );
        if !cost_gate.accepted {
            return Err(SemijoinRewriteRejection::with_analysis_and_cost_gate(
                SemijoinReductionRejectedReason::CostGateRejected,
                analysis,
                cost_gate,
            ));
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

fn resolve_output_slot(schema: &ArrowSchemaRef, column: &ColumnRef) -> Result<usize, SemijoinReductionRejectedReason> {
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
    analysis: &SemijoinLineageAnalysis,
    threshold_multiplier: u64,
) -> SemijoinReductionCostGateDiagnostics {
    let target_subtree = analysis.target_subtree_id;
    let threshold_multiplier = threshold_multiplier.max(1);

    if let Some(signal) = filtered_source_keyset_cost_signal(plan, semi_right, analysis) {
        return SemijoinReductionCostGateDiagnostics { signal, threshold_multiplier, accepted: true };
    }

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
    if accepted {
        return SemijoinReductionCostGateDiagnostics {
            signal: SemijoinReductionCostSignal::AggregateKeysetShape,
            threshold_multiplier,
            accepted,
        };
    }

    SemijoinReductionCostGateDiagnostics {
        signal: SemijoinReductionCostSignal::Unavailable,
        threshold_multiplier,
        accepted: false,
    }
}

fn filtered_source_keyset_cost_signal(
    plan: &QueryPlan,
    semi_right: NodeId,
    analysis: &SemijoinLineageAnalysis,
) -> Option<SemijoinReductionCostSignal> {
    if analysis.steps.is_empty() {
        return None;
    }

    let keyset = filtered_source_keyset_evidence(plan, semi_right)?;
    if keyset.filtered_source_count == 0 {
        return None;
    }

    let target_rows = estimated_output_rows(plan, analysis.target_subtree_id);
    let target_bytes = estimated_output_bytes(plan, analysis.target_subtree_id);
    if !OptimizerSourceTargetPolicy::default().target_is_dimension_sized(target_rows, target_bytes) {
        return None;
    }

    Some(SemijoinReductionCostSignal::FilteredSourceKeysetShape {
        filtered_source_count: keyset.filtered_source_count,
        semi_right_source_bytes: keyset.source_bytes,
        target_rows,
        target_bytes,
        crossed_join_count: analysis.steps.len(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FilteredSourceKeysetEvidence {
    filtered_source_count: usize,
    source_bytes: Option<u64>,
}

impl FilteredSourceKeysetEvidence {
    fn source() -> Self {
        Self { filtered_source_count: 0, source_bytes: Some(0) }
    }

    fn filtered(mut self) -> Self {
        self.filtered_source_count = self.filtered_source_count.saturating_add(1);
        self
    }

    fn with_source_bytes(mut self, bytes: Option<u64>) -> Self {
        self.source_bytes = combine_optional_u64(self.source_bytes, bytes);
        self
    }

    fn combine(self, other: Self) -> Self {
        Self {
            filtered_source_count: self.filtered_source_count.saturating_add(other.filtered_source_count),
            source_bytes: combine_optional_u64(self.source_bytes, other.source_bytes),
        }
    }
}

fn filtered_source_keyset_evidence(plan: &QueryPlan, node_id: NodeId) -> Option<FilteredSourceKeysetEvidence> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => {
                let evidence =
                    FilteredSourceKeysetEvidence::source().with_source_bytes(estimated_output_bytes(plan, node_id));
                Some(if source.reader_predicate.is_some() { evidence.filtered() } else { evidence })
            },
            crate::source::SourceNode::LiteralBatch(source) => Some(
                FilteredSourceKeysetEvidence::source().with_source_bytes(
                    source
                        .batches
                        .iter()
                        .try_fold(0_u64, |sum, batch| sum.checked_add(batch.get_array_memory_size() as u64)),
                ),
            ),
            crate::source::SourceNode::GpuResident(_) => Some(FilteredSourceKeysetEvidence::source()),
        },
        PlanNode::Filter(filter) => Some(filtered_source_keyset_evidence(plan, filter.input)?.filtered()),
        PlanNode::Projection(projection) => filtered_source_keyset_evidence(plan, projection.input),
        PlanNode::Sort(sort) => filtered_source_keyset_evidence(plan, sort.input),
        PlanNode::Limit(limit) => filtered_source_keyset_evidence(plan, limit.input),
        PlanNode::Join(join)
            if join.spec.join_type == JoinType::Inner
                && join.spec.keys.len() == 1
                && join.spec.residual_predicate.is_none()
                && join.spec.build_artifact_id.is_none() =>
        {
            let left = filtered_source_keyset_evidence(plan, join.inputs[0])?;
            let right = filtered_source_keyset_evidence(plan, join.inputs[1])?;
            Some(left.combine(right))
        },
        PlanNode::Join(join)
            if join.spec.join_type == JoinType::LeftSemi
                && join.spec.keys.len() == 1
                && join.spec.residual_predicate.is_none()
                && join.spec.build_artifact_id.is_none()
                && join.spec.null_equality == JoinNullEquality::NullsDoNotMatch
                && join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left) =>
        {
            let left = filtered_source_keyset_evidence(plan, join.inputs[0])?;
            let right = filtered_source_keyset_evidence(plan, join.inputs[1])?;
            Some(left.combine(right))
        },
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

fn combine_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => left.checked_add(right),
        _ => None,
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
        PlanNode::Source(_)
        | PlanNode::Join(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
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
        PlanNode::Aggregate(_)
        | PlanNode::Join(_)
        | PlanNode::Window(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
    }
}

fn left_output_projection_for_schema(schema: &ArrowSchemaRef) -> Vec<JoinOutputColumn> {
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

fn semijoin_projection_schema(plan: &QueryPlan, join: &JoinNode) -> ArrowSchemaRef {
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

fn schemas_equal(left: &ArrowSchemaRef, right: &ArrowSchemaRef) -> bool {
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
    let keep = matches!((original_admitted, rewritten_admitted), (Some(_), Some(true)) | (Some(false), Some(false)));
    if keep {
        AdmissionGateDecision::Keep { decision: kept, original_admitted, rewritten_admitted }
    } else {
        AdmissionGateDecision::Reject { decision: rejected, original_admitted, rewritten_admitted }
    }
}

fn strict_bounded_admitted(plan: &QueryPlan) -> Option<bool> {
    match crate::admission::strict_bounded_pipeline_admission(plan) {
        Ok(Some(_)) => Some(true),
        Ok(None) => Some(false),
        Err(error) if error.code() == crate::error::QueryErrorCode::UnsupportedPlanShape => Some(false),
        Err(_) => None,
    }
}

fn adaptive_admitted(plan: &QueryPlan) -> Option<bool> {
    match crate::admission::adaptive_bounded_pipeline_admission(plan) {
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

#[cfg(test)]
#[path = "semijoin_reduction_tests.rs"]
mod tests;
