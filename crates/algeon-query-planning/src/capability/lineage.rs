//! Output-field lineage used by invariant proofs.
//!
//! A lineage names the field a value came from, never the value itself. Two
//! lineages compare equal when they name the same source field, the same
//! aggregate group or measure, or the same derived field of the same node;
//! that equality is what the shared-producer invariants test. Every failure
//! is a [`CapabilityDecision`] rejection so the caller can report it as the
//! invariant's own reason.

use super::{
    AggregateFunction, Expr, NodeId, PlanNode, QueryPlan, expr::expr_family, graph::join_side_index,
    model::CapabilityDecision, rejection::Reason,
};
use crate::expr::ColumnRef;

/// Identity of one output field; see [`output_lineage`] for how each operator resolves it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum OutputLineage {
    /// A field read from a source node.
    SourceField { source_id: NodeId, field_index: usize },
    /// A grouping key of a grouped aggregate.
    AggregateGroup { aggregate_id: NodeId, group_index: usize },
    /// A measure of a grouped aggregate, identified by the node alone because
    /// each output row is a group and no single input field identifies it.
    AggregateMeasure { aggregate_id: NodeId, aggr_index: usize, function: AggregateFunction },
    /// A measure of a scalar aggregate, which keeps the lineage of the field it
    /// aggregates so a consumer can check which grouped measure it reads.
    ScalarAggregateMeasure {
        aggregate_id: NodeId,
        aggr_index: usize,
        function: AggregateFunction,
        input: Box<OutputLineage>,
    },
    /// A field computed by `node_id` that resolves to no single input field.
    Derived { node_id: NodeId, field_index: usize },
}

fn column_expr_lineage(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
    reason: Reason,
) -> Result<OutputLineage, CapabilityDecision> {
    match expr {
        Expr::Column(column) => {
            let field_index = resolve_column_field_index(plan, input_id, column, reason)?;
            output_lineage(plan, input_id, field_index)
        },
        Expr::Alias { expr, .. } => column_expr_lineage(plan, input_id, expr, reason),
        _ => Err(CapabilityDecision::rejected(reason, [("expression_family", expr_family(expr).to_string())])),
    }
}

fn expression_lineage_for_input(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
) -> Result<OutputLineage, CapabilityDecision> {
    column_expr_lineage(plan, input_id, expr, Reason::ExpressionLineageRequiresResolvedInputColumn)
}

/// Lineages of the fields that `expr`, evaluated over the output of
/// `input_id`, reads.
///
/// A projection output is a row-wise function of the fields its expression
/// reads, so reading it reads those fields. The walk descends through every
/// projection and resolves the remaining columns with [`output_lineage`]; a
/// literal reads no field.
pub(super) fn expression_read_lineages(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
) -> Result<Vec<OutputLineage>, CapabilityDecision> {
    let mut lineages = Vec::new();
    for column in expr.columns() {
        let field_index =
            resolve_column_field_index(plan, input_id, column, Reason::ExpressionLineageRequiresResolvedInputColumn)?;
        match plan.node(input_id).map(|node| &node.kind) {
            Some(PlanNode::Projection(projection)) => {
                let Some(output) = projection.outputs.get(field_index) else {
                    return Err(CapabilityDecision::rejected(
                        Reason::OutputLineageProjectionFieldIndexOutOfRange,
                        [("node_id", input_id.to_string()), ("field_index", field_index.to_string())],
                    ));
                };
                lineages.extend(expression_read_lineages(plan, projection.input, &output.expr)?);
            },
            _ => lineages.push(output_lineage(plan, input_id, field_index)?),
        }
    }
    Ok(lineages)
}

/// Resolves the identity of one output field of `node_id`.
///
/// Lineage records which field a value came from and says nothing about the
/// value itself; two lineages compare equal when they name the same source
/// field, the same aggregate group or measure, or the same derived field of
/// the same node. Filter, sort, and limit forward their input's lineage
/// unchanged. A projection or join output that is a bare or aliased column
/// reference resolves to the input field it selects, and a window
/// pass-through column resolves to the window input. A projection output that
/// aliases any other expression is rejected as unresolved. Every other
/// projection expression, every window-computed field, and every union,
/// edge-normalize, graph, and vector output is `Derived` under its own node
/// and field index.
///
/// A scalar aggregate (no group expressions) retains its input's lineage so a
/// consumer can check that a scalar measure reads one specific grouped
/// measure. A grouped measure is identified by its aggregate node alone,
/// because each of its output rows is a group and no single input field
/// identifies it.
pub(super) fn output_lineage(
    plan: &QueryPlan,
    node_id: NodeId,
    field_index: usize,
) -> Result<OutputLineage, CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            Reason::OutputLineageMissingNode,
            [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
        ));
    };
    if field_index >= node.output_schema().fields().len() {
        return Err(CapabilityDecision::rejected(
            Reason::OutputLineageFieldIndexOutOfRange,
            [
                ("node_id", node_id.to_string()),
                ("field_index", field_index.to_string()),
                ("field_count", node.output_schema().fields().len().to_string()),
            ],
        ));
    }
    match &node.kind {
        PlanNode::Source(_) => Ok(OutputLineage::SourceField { source_id: node_id, field_index }),
        PlanNode::Filter(filter) => output_lineage(plan, filter.input, field_index),
        PlanNode::Sort(sort) => output_lineage(plan, sort.input, field_index),
        PlanNode::Limit(limit) => output_lineage(plan, limit.input, field_index),
        PlanNode::Projection(projection) => {
            let Some(output) = projection.outputs.get(field_index) else {
                return Err(CapabilityDecision::rejected(
                    Reason::OutputLineageProjectionFieldIndexOutOfRange,
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            match &output.expr {
                Expr::Column(column) => {
                    let input_field = resolve_column_field_index(
                        plan,
                        projection.input,
                        column,
                        Reason::OutputLineageProjectionColumnResolution,
                    )?;
                    output_lineage(plan, projection.input, input_field)
                },
                Expr::Alias { expr, .. } => expression_lineage_for_input(plan, projection.input, expr),
                _ => Ok(OutputLineage::Derived { node_id, field_index }),
            }
        },
        PlanNode::Aggregate(aggregate) => {
            let group_count = aggregate.spec.group_exprs.len();
            if field_index < group_count {
                return Ok(OutputLineage::AggregateGroup { aggregate_id: node_id, group_index: field_index });
            }
            let aggr_index = field_index - group_count;
            let Some(aggregate_expr) = aggregate.spec.aggr_exprs.get(aggr_index) else {
                return Err(CapabilityDecision::rejected(
                    Reason::OutputLineageAggregateFieldIndexOutOfRange,
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            if group_count == 0 {
                let input = Box::new(expression_lineage_for_input(plan, aggregate.input, &aggregate_expr.expr)?);
                Ok(OutputLineage::ScalarAggregateMeasure {
                    aggregate_id: node_id,
                    aggr_index,
                    function: aggregate_expr.function,
                    input,
                })
            } else {
                Ok(OutputLineage::AggregateMeasure {
                    aggregate_id: node_id,
                    aggr_index,
                    function: aggregate_expr.function,
                })
            }
        },
        PlanNode::Join(join) => {
            let Some(output) = join.spec.output_projection.get(field_index) else {
                return Err(CapabilityDecision::rejected(
                    Reason::OutputLineageJoinProjectionIndexOutOfRange,
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            let input_id = join.inputs[join_side_index(output.side)];
            let input_field = resolve_column_field_index(
                plan,
                input_id,
                &output.column,
                Reason::OutputLineageJoinOutputColumnResolution,
            )?;
            output_lineage(plan, input_id, input_field)
        },
        PlanNode::Window(window) => {
            let input_field_count =
                plan.node(window.input).map(|input| input.output_schema().fields().len()).ok_or_else(|| {
                    CapabilityDecision::rejected(
                        Reason::OutputLineageWindowMissingInput,
                        [("node_id", node_id.to_string()), ("input_id", window.input.to_string())],
                    )
                })?;
            if field_index < input_field_count {
                output_lineage(plan, window.input, field_index)
            } else {
                Ok(OutputLineage::Derived { node_id, field_index })
            }
        },
        PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => Ok(OutputLineage::Derived { node_id, field_index }),
    }
}

/// Position of `column` in the output schema of `input_id`, matched by name
/// only. A name that matches no field or more than one field is rejected under
/// the caller's `reason`, with a `resolution` fact that says which.
fn resolve_column_field_index(
    plan: &QueryPlan,
    input_id: NodeId,
    column: &ColumnRef,
    reason: Reason,
) -> Result<usize, CapabilityDecision> {
    let Some(node) = plan.node(input_id) else {
        return Err(CapabilityDecision::rejected(reason, [("input_id", input_id.to_string())]));
    };
    let mut matches = node
        .output_schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| (field.name() == column.name()).then_some(index));
    let Some(index) = matches.next() else {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("input_id", input_id.to_string()),
                ("field_count", node.output_schema().fields().len().to_string()),
                ("resolution", "missing".to_string()),
            ],
        ));
    };
    if matches.next().is_some() {
        return Err(CapabilityDecision::rejected(
            reason,
            [
                ("input_id", input_id.to_string()),
                ("field_count", node.output_schema().fields().len().to_string()),
                ("resolution", "ambiguous".to_string()),
            ],
        ));
    }
    Ok(index)
}
