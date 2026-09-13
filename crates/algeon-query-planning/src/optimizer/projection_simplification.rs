use std::collections::{BTreeMap, BTreeSet};

use arrow::datatypes::SchemaRef;

use super::rewrite;
use crate::{
    capability::capability_shape_contract_projection_ids,
    expr::{ColumnRef, Expr},
    plan::{NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan},
};

/// Stable rule name used in optimizer diagnostics.
pub const PROJECTION_SIMPLIFICATION_RULE_NAME: &str = "projection_simplification";

/// Configuration for removing redundant identity projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionSimplificationOptions {
    /// Whether the pass is allowed to remove eligible projections.
    pub enabled: bool,
}

impl ProjectionSimplificationOptions {
    /// Returns options with the pass disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for ProjectionSimplificationOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for projection-simplification candidates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectionSimplificationDiagnostics {
    /// Number of projection nodes considered.
    pub considered_count: usize,
    /// Number of identity projections removed.
    pub applied_count: usize,
    /// Per-candidate evidence in traversal order.
    pub candidates: Vec<ProjectionSimplificationCandidateDiagnostics>,
}

impl ProjectionSimplificationDiagnostics {
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

/// Per-candidate projection-simplification evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionSimplificationCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Projection node considered before rewrite.
    pub candidate_node_id: NodeId,
    /// Input node that replaces the projection when applied.
    pub replacement_node_id: Option<NodeId>,
    /// True when this candidate was removed from the returned plan.
    pub applied: bool,
    /// Rejection reason when the projection was not removed.
    pub rejected_reason: Option<ProjectionSimplificationRejectedReason>,
}

/// Reasons an identity-projection simplification candidate was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionSimplificationRejectedReason {
    /// Pass was disabled.
    PassDisabled,
    /// Projection output schema differs from the input schema.
    SchemaMismatch,
    /// Projection contains a non-column expression or changes column mapping.
    NonIdentityExpression,
    /// Projection is part of a bounded shape contract.
    ShapeContract,
}

#[doc(hidden)]
pub struct ProjectionSimplificationResult {
    #[doc(hidden)]
    pub plan: QueryPlan,
    #[doc(hidden)]
    pub change: super::NativeOptimizerChange,
    #[doc(hidden)]
    pub diagnostics: ProjectionSimplificationDiagnostics,
}

#[doc(hidden)]
pub fn optimize_projection_simplification(
    plan: QueryPlan,
    options: &ProjectionSimplificationOptions,
) -> ProjectionSimplificationResult {
    if !options.enabled {
        return ProjectionSimplificationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: ProjectionSimplificationDiagnostics::default(),
        };
    }

    let mut rewriter = ProjectionSimplificationRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let diagnostics = rewriter.diagnostics.clone();
        return ProjectionSimplificationResult { plan, change: super::NativeOptimizerChange::Unchanged, diagnostics };
    }

    ProjectionSimplificationResult {
        plan: rewrite::finish_rewritten_plan(rewriter.nodes, new_root, plan.sink())
            .with_keyed_hash_join_strategy_options_from(&plan)
            .with_advisory_outputs_from(&plan),
        change: super::NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    }
}

struct ProjectionSimplificationRewriter<'a> {
    original: &'a QueryPlan,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    protected_projection_ids: BTreeSet<NodeId>,
    diagnostics: ProjectionSimplificationDiagnostics,
    changed: bool,
}

impl<'a> ProjectionSimplificationRewriter<'a> {
    fn new(original: &'a QueryPlan) -> Self {
        Self {
            original,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            protected_projection_ids: capability_shape_contract_projection_ids(original),
            diagnostics: ProjectionSimplificationDiagnostics::default(),
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if let Some(new_id) = self.memo.get(&old_id).copied() {
            return new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        if let PlanNode::Projection(projection) = &old_node.kind
            && let Some(replacement) = self.try_simplify_projection(old_id, projection)
        {
            self.memo.insert(old_id, replacement);
            return replacement;
        }

        let kind = self.rewrite_node_kind(&old_node.kind);
        let new_id = self.push_node(kind, old_node.facts.clone());
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn try_simplify_projection(&mut self, old_id: NodeId, projection: &ProjectionNode) -> Option<NodeId> {
        let mut candidate = ProjectionSimplificationCandidateDiagnostics {
            rule_name: PROJECTION_SIMPLIFICATION_RULE_NAME,
            candidate_node_id: old_id,
            replacement_node_id: None,
            applied: false,
            rejected_reason: None,
        };
        self.diagnostics.considered_count += 1;

        if self.protected_projection_ids.contains(&old_id) {
            candidate.rejected_reason = Some(ProjectionSimplificationRejectedReason::ShapeContract);
            self.diagnostics.candidates.push(candidate);
            return None;
        }

        let input_node = self.original.node(projection.input).expect("projection input must exist");
        if schemas_equal(input_node.output_schema(), projection.output_schema())
            && projection_is_identity(input_node.output_schema(), projection)
        {
            let replacement = self.rewrite_subtree(projection.input);
            candidate.replacement_node_id = Some(replacement);
            candidate.applied = true;
            self.diagnostics.applied_count += 1;
            self.diagnostics.candidates.push(candidate);
            self.changed = true;
            return Some(replacement);
        }

        if !self.protected_projection_ids.contains(&input_node.id)
            && let Some(fused_outputs) = compose_direct_projection_chain(input_node, projection)
        {
            let PlanNode::Projection(input_projection) = &input_node.kind else {
                unreachable!("compose_direct_projection_chain requires a projection input");
            };
            let fused_input = self.rewrite_subtree(input_projection.input);
            let facts = self.original.node(old_id).expect("projection node must exist").facts.clone();
            let replacement =
                self.push_node(PlanNode::Projection(ProjectionNode::new(fused_input, fused_outputs)), facts);
            candidate.replacement_node_id = Some(replacement);
            candidate.applied = true;
            self.diagnostics.applied_count += 1;
            self.diagnostics.candidates.push(candidate);
            self.changed = true;
            return Some(replacement);
        }

        candidate.rejected_reason = Some(if schemas_equal(input_node.output_schema(), projection.output_schema()) {
            ProjectionSimplificationRejectedReason::NonIdentityExpression
        } else {
            ProjectionSimplificationRejectedReason::SchemaMismatch
        });
        self.diagnostics.candidates.push(candidate);
        None
    }

    fn rewrite_node_kind(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}

fn projection_is_identity(input_schema: &SchemaRef, projection: &ProjectionNode) -> bool {
    projection.outputs.len() == input_schema.fields().len()
        && projection.outputs.iter().zip(input_schema.fields().iter()).all(|(output, input_field)| {
            output.field == input_field.as_ref().clone()
                && direct_column_name(&output.expr).is_some_and(|name| name == input_field.name())
        })
}

fn compose_direct_projection_chain(
    input_node: &QueryNode,
    projection: &ProjectionNode,
) -> Option<Vec<ProjectionOutput>> {
    let PlanNode::Projection(input_projection) = &input_node.kind else {
        return None;
    };
    let mut direct_outputs = BTreeMap::<String, Expr>::new();
    for output in &input_projection.outputs {
        direct_outputs.insert(output.field.name().clone(), direct_projection_source_expr(&output.expr)?);
    }

    projection
        .outputs
        .iter()
        .map(|output| {
            let expr = compose_direct_projection_expr(&output.expr, &direct_outputs)?;
            Some(ProjectionOutput::new(expr, output.field.clone()))
        })
        .collect()
}

fn compose_direct_projection_expr(expr: &Expr, direct_outputs: &BTreeMap<String, Expr>) -> Option<Expr> {
    match expr {
        Expr::Column(column) => direct_outputs.get(unqualified_column_name(column)?).cloned(),
        Expr::Alias { expr, alias } => Some(compose_direct_projection_expr(expr, direct_outputs)?.alias(alias.clone())),
        _ => None,
    }
}

fn direct_projection_source_expr(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Column(column) => Some(Expr::Column(column.clone())),
        Expr::Alias { expr, .. } => direct_projection_source_expr(expr),
        _ => None,
    }
}

fn direct_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => unqualified_column_name(column),
        Expr::Alias { expr, alias } => {
            let column = direct_column_name(expr)?;
            (column == alias).then_some(column)
        },
        _ => None,
    }
}

fn unqualified_column_name(column: &ColumnRef) -> Option<&str> {
    column.qualifier().is_none().then_some(column.name())
}

fn schemas_equal(left: &SchemaRef, right: &SchemaRef) -> bool {
    left.fields().len() == right.fields().len()
        && left.fields().iter().zip(right.fields().iter()).all(|(left, right)| left == right)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{BinaryOp, LiteralValue},
        plan::{ProjectionOutput, QueryPlanBuilder},
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn removes_chained_identity_projections() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", schema(vec![field("k"), field("v")])));
        let first = builder.add_projection(source, identity_outputs(&["k", "v"]));
        let second = builder.add_projection(first, identity_alias_outputs(&["k", "v"]));
        let plan = builder.finish_gpu_resident(second);

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default());

        assert!(result.change.changed());
        assert_eq!(result.diagnostics.considered_count, 2);
        assert_eq!(result.diagnostics.applied_count, 2);
        assert_eq!(result.plan.node_count(), 1);
        assert!(matches!(result.plan.root_node().kind, PlanNode::Source(_)));
        assert!(result.plan.unreachable_node_ids().is_empty());
    }

    #[test]
    fn keeps_projection_that_changes_output_name() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", schema(vec![field("k")])));
        let projection = builder
            .add_projection(source, vec![ProjectionOutput::new(Expr::column("k").alias("renamed"), field("renamed"))]);
        let plan = builder.finish_gpu_resident(projection);

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default());

        assert!(!result.change.changed());
        assert_eq!(result.diagnostics.considered_count, 1);
        assert_eq!(result.diagnostics.applied_count, 0);
        assert_eq!(
            result.diagnostics.candidates[0].rejected_reason,
            Some(ProjectionSimplificationRejectedReason::SchemaMismatch)
        );
    }

    #[test]
    fn keeps_projection_with_computed_expression() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", schema(vec![field("k")])));
        let projection = builder.add_projection(
            source,
            vec![ProjectionOutput::new(
                Expr::binary(Expr::column("k"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(1))).alias("k"),
                field("k"),
            )],
        );
        let plan = builder.finish_gpu_resident(projection);

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default());

        assert!(!result.change.changed());
        assert_eq!(result.diagnostics.considered_count, 1);
        assert_eq!(result.diagnostics.applied_count, 0);
        assert_eq!(
            result.diagnostics.candidates[0].rejected_reason,
            Some(ProjectionSimplificationRejectedReason::NonIdentityExpression)
        );
    }

    #[test]
    fn fuses_direct_projection_chain() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", schema(vec![field("k"), field("v"), field("payload")])));
        let first = builder.add_projection(source, identity_alias_outputs(&["k", "v", "payload"]));
        let second = builder
            .add_projection(first, vec![ProjectionOutput::new(Expr::column("v").alias("renamed"), field("renamed"))]);
        let plan = builder.finish_gpu_resident(second);

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default());

        assert!(result.change.changed());
        assert_eq!(result.diagnostics.considered_count, 1);
        assert_eq!(result.diagnostics.applied_count, 1);
        assert_eq!(result.plan.node_count(), 2);
        let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
            panic!("root should remain a projection");
        };
        assert!(matches!(result.plan.node(projection.input).map(|node| &node.kind), Some(PlanNode::Source(_))));
        assert_eq!(projection.outputs.len(), 1);
        assert_eq!(projection.outputs[0].field.name(), "renamed");
        assert_eq!(projection.outputs[0].expr.to_string(), "v AS renamed");
        assert!(result.plan.unreachable_node_ids().is_empty());
    }

    #[test]
    fn keeps_projection_chain_when_inner_projection_computes_expression() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", schema(vec![field("k")])));
        let first = builder.add_projection(
            source,
            vec![ProjectionOutput::new(
                Expr::binary(Expr::column("k"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(1))).alias("computed"),
                field("computed"),
            )],
        );
        let second =
            builder.add_projection(first, vec![ProjectionOutput::new(Expr::column("computed"), field("renamed"))]);
        let plan = builder.finish_gpu_resident(second);

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default());

        assert!(!result.change.changed());
        assert_eq!(result.diagnostics.considered_count, 2);
        assert_eq!(result.diagnostics.applied_count, 0);
        assert!(
            result
                .diagnostics
                .candidates
                .iter()
                .any(|candidate| candidate.rejected_reason
                    == Some(ProjectionSimplificationRejectedReason::SchemaMismatch))
        );
        assert!(result.diagnostics.candidates.iter().all(|candidate| !candidate.applied));
    }

    fn source(name: &str, schema: Arc<Schema>) -> SourceNode {
        SourceNode::GpuResident(GpuResidentSource::new(name, schema))
    }

    fn schema(fields: Vec<Field>) -> Arc<Schema> {
        Arc::new(Schema::new(fields))
    }

    fn field(name: &str) -> Field {
        Field::new(name, DataType::Int32, false)
    }

    fn identity_outputs(names: &[&str]) -> Vec<ProjectionOutput> {
        names.iter().map(|name| ProjectionOutput::new(Expr::column(*name), field(name))).collect()
    }

    fn identity_alias_outputs(names: &[&str]) -> Vec<ProjectionOutput> {
        names
            .iter()
            .map(|name| ProjectionOutput::new(Expr::column(*name).alias((*name).to_string()), field(name)))
            .collect()
    }
}
