#![allow(dead_code)]

use crate::{
    expr::{ColumnQualifier, ColumnRef, Expr},
    plan::{JoinKey, JoinSide, JoinType, NodeId, PlanNode, QueryPlan},
};

pub const SEMIJOIN_REDUCTION_RULE_NAME: &str = "semijoin_reduction";

/// Configuration for the semijoin-reduction optimizer pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemijoinReductionOptions {
    pub enabled: bool,
}

impl SemijoinReductionOptions {
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for SemijoinReductionOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// No-op Phase 1 result for the semijoin-reduction pass.
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
    CostGateRejected,
    AdmissionRejectedRewrite,
}

/// Runs the semijoin-reduction pass.
///
/// Phase 1 deliberately exposes the pass surface and diagnostics while leaving
/// the plan unchanged. Later phases will populate candidate diagnostics and
/// apply rewrites behind this stable entry point.
pub fn optimize_semijoin_reduction(plan: QueryPlan, _options: &SemijoinReductionOptions) -> SemijoinReductionResult {
    SemijoinReductionResult {
        plan,
        change: super::NativeOptimizerChange::Unchanged,
        diagnostics: SemijoinReductionDiagnostics::default(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SemijoinLineageAnalysis {
    pub target_subtree_id: NodeId,
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
        Ok(SemijoinLineageAnalysis { target_subtree_id: node_id, key_mappings, steps: Vec::new() })
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateSpec, BinaryOp, LimitSpec, SortExpr, SortSpec},
        plan::{JoinOutputColumn, JoinSpec, ProjectionOutput, QueryPlanBuilder},
        source::{GpuResidentSource, SourceNode},
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
