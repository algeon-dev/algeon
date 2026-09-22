//! Output-field lineage used by invariant proofs.

use super::{
    AggregateFunction, Expr, NodeId, PlanNode, QueryPlan, expr::expr_family, graph::join_side_index,
    model::CapabilityDecision,
};
use crate::expr::ColumnRef;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum OutputLineage {
    SourceField {
        source_id: NodeId,
        field_index: usize,
    },
    AggregateGroup {
        aggregate_id: NodeId,
        group_index: usize,
    },
    AggregateMeasure {
        aggregate_id: NodeId,
        aggr_index: usize,
        function: AggregateFunction,
    },
    ScalarAggregateMeasure {
        aggregate_id: NodeId,
        aggr_index: usize,
        function: AggregateFunction,
        input: Box<OutputLineage>,
    },
    Derived {
        node_id: NodeId,
        field_index: usize,
    },
}

fn column_expr_lineage(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
    reason: &'static str,
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

pub(super) fn expression_lineage_for_input(
    plan: &QueryPlan,
    input_id: NodeId,
    expr: &Expr,
) -> Result<OutputLineage, CapabilityDecision> {
    column_expr_lineage(plan, input_id, expr, "expression_lineage_requires_resolved_input_column")
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
            "output_lineage_missing_node",
            [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
        ));
    };
    if field_index >= node.output_schema().fields().len() {
        return Err(CapabilityDecision::rejected(
            "output_lineage_field_index_out_of_range",
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
                    "output_lineage_projection_field_index_out_of_range",
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            match &output.expr {
                Expr::Column(column) => {
                    let input_field = resolve_column_field_index(
                        plan,
                        projection.input,
                        column,
                        "output_lineage_projection_column_resolution",
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
                    "output_lineage_aggregate_field_index_out_of_range",
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
                    "output_lineage_join_projection_index_out_of_range",
                    [("node_id", node_id.to_string()), ("field_index", field_index.to_string())],
                ));
            };
            let input_id = join.inputs[join_side_index(output.side)];
            let input_field = resolve_column_field_index(
                plan,
                input_id,
                &output.column,
                "output_lineage_join_output_column_resolution",
            )?;
            output_lineage(plan, input_id, input_field)
        },
        PlanNode::Window(window) => {
            let input_field_count =
                plan.node(window.input).map(|input| input.output_schema().fields().len()).ok_or_else(|| {
                    CapabilityDecision::rejected(
                        "output_lineage_window_missing_input",
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

fn resolve_column_field_index(
    plan: &QueryPlan,
    input_id: NodeId,
    column: &ColumnRef,
    reason: &'static str,
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
