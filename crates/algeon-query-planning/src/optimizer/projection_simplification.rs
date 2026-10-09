//! Projection simplification: removes identity projections and fuses chains of
//! renaming projections.
//!
//! Pattern one: a projection whose outputs are, in order, every input field
//! under its own name and type. It is removed and its consumers read the input.
//! Pattern two: a projection over a projection whose outputs are all bare or
//! aliased columns; the outer column references are replaced by the inner
//! outputs' source columns and the inner projection disappears. Both keep every
//! row and every output field, so the facts at the consumer do not change. Only
//! unqualified names are matched; a qualified reference names a join input and
//! is left alone. The node list is rebuilt, so `QueryPlan::rewrite` renumbers
//! it and drops routes through removed nodes. The rule is safe to run more
//! than once; other rules introduce projections, and the pass table in
//! `optimizer.rs` decides where it runs.
//! Rejection reasons are report contracts; `SchemaMismatch` also covers a plain
//! rename, whose output field differs from the input field.

use std::collections::BTreeMap;

use arrow::datatypes::SchemaRef;

use super::rewrite;
use crate::{
    error::Result,
    expr::{ColumnRef, Expr},
    plan::{NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, RewrittenNodes},
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
    /// Number of projections replaced by their input or by a fused projection.
    pub applied_count: usize,
    /// Per-candidate evidence in traversal order.
    pub candidates: Vec<ProjectionSimplificationCandidateDiagnostics>,
}

impl ProjectionSimplificationDiagnostics {
    pub(crate) fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.applied_count == 0 && self.candidates.is_empty()
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.applied_count += other.applied_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate projection-simplification evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionSimplificationCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub(crate) rule_name: &'static str,
    /// Projection node considered before rewrite.
    pub(crate) candidate_node_id: NodeId,
    /// Node that replaces the projection when applied. It is the input's image for
    /// an identity projection and the fused projection for a chain.
    pub(crate) replacement_node_id: Option<NodeId>,
    /// True when this projection was replaced in the returned plan.
    pub(crate) applied: bool,
    /// Rejection reason when the projection was not replaced.
    pub rejected_reason: Option<ProjectionSimplificationRejectedReason>,
}

/// Reasons a projection-simplification candidate was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionSimplificationRejectedReason {
    /// Pass was disabled.
    PassDisabled,
    /// Projection output schema differs from the input schema.
    SchemaMismatch,
    /// Projection contains a non-column expression or changes column mapping.
    NonIdentityExpression,
}

pub(crate) struct ProjectionSimplificationResult {
    pub(crate) plan: QueryPlan,
    pub(crate) change: super::NativeOptimizerChange,
    pub(crate) diagnostics: ProjectionSimplificationDiagnostics,
}

pub(crate) fn optimize_projection_simplification(
    plan: QueryPlan,
    options: &ProjectionSimplificationOptions,
) -> Result<ProjectionSimplificationResult> {
    if !options.enabled {
        return Ok(ProjectionSimplificationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: ProjectionSimplificationDiagnostics::default(),
        });
    }

    let mut rewriter = ProjectionSimplificationRewriter::new(&plan);
    let new_root = rewriter.rewrite_subtree(plan.root_id());
    if !rewriter.changed {
        let diagnostics = rewriter.diagnostics.clone();
        return Ok(ProjectionSimplificationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics,
        });
    }

    Ok(ProjectionSimplificationResult {
        plan: plan.rewrite(RewrittenNodes::rebuilt(rewriter.nodes, new_root, rewriter.memo))?,
        change: super::NativeOptimizerChange::Changed,
        diagnostics: rewriter.diagnostics,
    })
}

/// Rebuilds the plan top down. `memo` maps each visited source node to its
/// image, so a shared subtree is rewritten once and a removed projection maps
/// to its input's image; `memo` is also the origin map passed to
/// `RewrittenNodes::rebuilt`.
struct ProjectionSimplificationRewriter<'a> {
    original: &'a QueryPlan,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    diagnostics: ProjectionSimplificationDiagnostics,
    changed: bool,
}

impl<'a> ProjectionSimplificationRewriter<'a> {
    fn new(original: &'a QueryPlan) -> Self {
        Self {
            original,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
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

        if let Some(fused_outputs) = compose_direct_projection_chain(input_node, projection) {
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
        let id = NodeId(self.nodes.len());
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

/// Outputs of `projection` restated over its input projection's input, when the
/// inner projection only renames columns. Any computed inner output makes the
/// chain opaque, since the outer reference would then name a value the fused
/// node could not read.
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
        let first = builder.add_projection(source, identity_outputs(&["k", "v"])).unwrap();
        let second = builder.add_projection(first, identity_alias_outputs(&["k", "v"])).unwrap();
        let plan = builder.finish_gpu_resident(second).unwrap();

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default()).unwrap();

        assert!(result.change.changed());
        assert_eq!(result.diagnostics.considered_count, 2);
        assert_eq!(result.diagnostics.applied_count, 2);
        assert_eq!(result.plan.node_count(), 1);
        assert!(matches!(result.plan.root_node().kind, PlanNode::Source(_)));
    }

    #[test]
    fn keeps_projection_that_changes_output_name() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", schema(vec![field("k")])));
        let projection = builder
            .add_projection(source, vec![ProjectionOutput::new(Expr::column("k").alias("renamed"), field("renamed"))])
            .unwrap();
        let plan = builder.finish_gpu_resident(projection).unwrap();

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default()).unwrap();

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
        let projection = builder
            .add_projection(
                source,
                vec![ProjectionOutput::new(
                    Expr::binary(Expr::column("k"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(1))).alias("k"),
                    field("k"),
                )],
            )
            .unwrap();
        let plan = builder.finish_gpu_resident(projection).unwrap();

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default()).unwrap();

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
        let first = builder.add_projection(source, identity_alias_outputs(&["k", "v", "payload"])).unwrap();
        let second = builder
            .add_projection(first, vec![ProjectionOutput::new(Expr::column("v").alias("renamed"), field("renamed"))])
            .unwrap();
        let plan = builder.finish_gpu_resident(second).unwrap();

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default()).unwrap();

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
    }

    #[test]
    fn keeps_projection_chain_when_inner_projection_computes_expression() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(source("input", schema(vec![field("k")])));
        let first = builder
            .add_projection(
                source,
                vec![ProjectionOutput::new(
                    Expr::binary(Expr::column("k"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(1)))
                        .alias("computed"),
                    field("computed"),
                )],
            )
            .unwrap();
        let second = builder
            .add_projection(first, vec![ProjectionOutput::new(Expr::column("computed"), field("renamed"))])
            .unwrap();
        let plan = builder.finish_gpu_resident(second).unwrap();

        let result = optimize_projection_simplification(plan, &ProjectionSimplificationOptions::default()).unwrap();

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
