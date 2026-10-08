//! Column lineage below an aggregate: whether a column can be carried from a
//! source up to the aggregate input, and from which join input.
//!
//! The `materialized` variants also accept a column a widened source would
//! add; the plain variants see only what the plan reads today. A join can
//! carry a column only when one input and not the other can, since the
//! rewrite must know which side to extend.

use std::{collections::BTreeMap, sync::Arc};

use arrow::datatypes::{Field, Schema};

use super::{
    expr_helpers::field_with_name,
    signature::{
        MaterializedColumnLineage, SourceColumnMaterialization, local_source_materialization_signature,
        source_column_ref_signature,
    },
    source_widening::matching_source_entry,
};
use crate::{
    expr::Expr,
    plan::{JoinOutputColumn, JoinSide, NodeId, PlanNode, ProjectionOutput, QueryPlan},
    source::SourceNode,
};

pub(super) fn unique_join_input_that_can_propagate_column(
    plan: &QueryPlan,
    left: NodeId,
    right: NodeId,
    column: &str,
) -> Option<JoinSide> {
    match (subtree_can_propagate_column(plan, left, column), subtree_can_propagate_column(plan, right, column)) {
        (true, false) => Some(JoinSide::Left),
        (false, true) => Some(JoinSide::Right),
        _ => None,
    }
}

pub(super) fn unique_join_input_that_can_propagate_materialized_column(
    plan: &QueryPlan,
    left: NodeId,
    right: NodeId,
    column: &str,
    widened_source_columns: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> Option<JoinSide> {
    match (
        subtree_can_propagate_materialized_column(plan, left, column, widened_source_columns),
        subtree_can_propagate_materialized_column(plan, right, column, widened_source_columns),
    ) {
        (true, false) => Some(JoinSide::Left),
        (false, true) => Some(JoinSide::Right),
        _ => None,
    }
}

fn subtree_can_propagate_column(plan: &QueryPlan, node_id: NodeId, column: &str) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    if field_with_name(node.output_schema(), column).is_some() {
        return true;
    }
    match &node.kind {
        PlanNode::Source(_) => false,
        PlanNode::Filter(filter) => subtree_can_propagate_column(plan, filter.input, column),
        PlanNode::Projection(projection) => subtree_can_propagate_column(plan, projection.input, column),
        PlanNode::Join(join) => {
            unique_join_input_that_can_propagate_column(plan, join.inputs[0], join.inputs[1], column).is_some()
        },
        PlanNode::Sort(sort) => subtree_can_propagate_column(plan, sort.input, column),
        PlanNode::Limit(limit) => subtree_can_propagate_column(plan, limit.input, column),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
    }
}

pub(super) fn subtree_can_propagate_materialized_column(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    widened_source_columns: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                if field_with_name(&source.schema, column).is_some() {
                    return true;
                }
                let signature = local_source_materialization_signature(source);
                matching_source_entry(widened_source_columns, &signature)
                    .is_some_and(|columns| columns.contains_key(column))
            },
            _ => false,
        },
        PlanNode::Filter(filter) => {
            subtree_can_propagate_materialized_column(plan, filter.input, column, widened_source_columns)
        },
        PlanNode::Projection(projection) => {
            if let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column) {
                return projection_output_preserves_column(output, column);
            }
            subtree_can_propagate_materialized_column(plan, projection.input, column, widened_source_columns)
        },
        PlanNode::Join(join) => unique_join_input_that_can_propagate_materialized_column(
            plan,
            join.inputs[0],
            join.inputs[1],
            column,
            widened_source_columns,
        )
        .is_some(),
        PlanNode::Sort(sort) => {
            subtree_can_propagate_materialized_column(plan, sort.input, column, widened_source_columns)
        },
        PlanNode::Limit(limit) => {
            subtree_can_propagate_materialized_column(plan, limit.input, column, widened_source_columns)
        },
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
    }
}

/// Source identity of a column argument: rowset fingerprint plus the file or
/// partition reference it reads. Two members whose arguments share one name
/// but resolve to different sources must not be combined.
pub(super) fn scalar_input_expr_lineage_signature(plan: &QueryPlan, node_id: NodeId, expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => scalar_input_column_lineage_signature(plan, node_id, column.name()),
        Expr::Alias { expr, .. } => scalar_input_expr_lineage_signature(plan, node_id, expr),
        _ => None,
    }
}

pub(super) fn materialized_column_lineage(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
) -> Option<MaterializedColumnLineage> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                let signature = local_source_materialization_signature(source);
                let column = signature.columns.get(column)?;
                Some(MaterializedColumnLineage {
                    source_fingerprint: signature.abstract_fingerprint,
                    source_column: column.name.clone(),
                })
            },
            _ => None,
        },
        PlanNode::Filter(filter) => materialized_column_lineage(plan, filter.input, column),
        PlanNode::Projection(projection) => {
            let output = projection.outputs.iter().find(|output| output.field.name() == column)?;
            materialized_expr_lineage(plan, projection.input, &output.expr)
        },
        PlanNode::Join(join) => {
            let output = join.spec.output_projection.iter().find(|output| output.output_name == column)?;
            let input = match output.side {
                JoinSide::Left => join.inputs[0],
                JoinSide::Right => join.inputs[1],
            };
            materialized_column_lineage(plan, input, output.column.name())
        },
        PlanNode::Sort(sort) => materialized_column_lineage(plan, sort.input, column),
        PlanNode::Limit(limit) => materialized_column_lineage(plan, limit.input, column),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

fn materialized_expr_lineage(plan: &QueryPlan, node_id: NodeId, expr: &Expr) -> Option<MaterializedColumnLineage> {
    match expr {
        Expr::Column(column) => materialized_column_lineage(plan, node_id, column.name()),
        Expr::Alias { expr, .. } => materialized_expr_lineage(plan, node_id, expr),
        _ => None,
    }
}

/// Whether the representative's input exposes a column under the member's
/// argument name without a source lineage for it. The member's argument
/// would then bind to that unrelated column in the shared aggregate.
pub(super) fn scalar_input_expr_name_collides_with_unresolved_output(
    plan: &QueryPlan,
    node_id: NodeId,
    expr: &Expr,
) -> bool {
    match expr {
        Expr::Column(column) => {
            let Some(node) = plan.node(node_id) else {
                return true;
            };
            field_with_name(node.output_schema(), column.name()).is_some()
                && scalar_input_column_lineage_signature(plan, node_id, column.name()).is_none()
        },
        Expr::Alias { expr, .. } => scalar_input_expr_name_collides_with_unresolved_output(plan, node_id, expr),
        _ => false,
    }
}

fn scalar_input_column_lineage_signature(plan: &QueryPlan, node_id: NodeId, column: &str) -> Option<String> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                field_with_name(&source.schema, column)?;
                let signature = local_source_materialization_signature(source);
                let source_ref = source.source_column_ref(column)?;
                Some(format!(
                    "{};column={column};source_ref={}",
                    signature.rowset_fingerprint,
                    source_column_ref_signature(source_ref)
                ))
            },
            _ => None,
        },
        PlanNode::Filter(filter) => scalar_input_column_lineage_signature(plan, filter.input, column),
        PlanNode::Projection(projection) => {
            if let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column) {
                return scalar_input_expr_lineage_signature(plan, projection.input, &output.expr);
            }
            scalar_input_column_lineage_signature(plan, projection.input, column)
        },
        PlanNode::Sort(sort) => scalar_input_column_lineage_signature(plan, sort.input, column),
        PlanNode::Limit(limit) => scalar_input_column_lineage_signature(plan, limit.input, column),
        PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

pub(super) fn propagatable_materialized_field(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    widened_source_columns: &BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
) -> Option<Field> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                if let Some(field) = field_with_name(&source.schema, column) {
                    return Some(field.clone());
                }
                let signature = local_source_materialization_signature(source);
                matching_source_entry(widened_source_columns, &signature)
                    .and_then(|columns| columns.get(column))
                    .map(|column| column.field.clone())
            },
            _ => None,
        },
        PlanNode::Filter(filter) => propagatable_materialized_field(plan, filter.input, column, widened_source_columns),
        PlanNode::Projection(projection) => {
            if let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column) {
                return projection_output_preserves_column(output, column).then(|| output.field.clone());
            }
            propagatable_materialized_field(plan, projection.input, column, widened_source_columns)
        },
        PlanNode::Join(join) => {
            if let Some(output) = join.spec.output_projection.iter().find(|output| output.output_name == column) {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                return propagatable_materialized_field(plan, input, output.column.name(), widened_source_columns);
            }
            match unique_join_input_that_can_propagate_materialized_column(
                plan,
                join.inputs[0],
                join.inputs[1],
                column,
                widened_source_columns,
            )? {
                JoinSide::Left => propagatable_materialized_field(plan, join.inputs[0], column, widened_source_columns),
                JoinSide::Right => {
                    propagatable_materialized_field(plan, join.inputs[1], column, widened_source_columns)
                },
            }
        },
        PlanNode::Sort(sort) => propagatable_materialized_field(plan, sort.input, column, widened_source_columns),
        PlanNode::Limit(limit) => propagatable_materialized_field(plan, limit.input, column, widened_source_columns),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

pub(super) fn projection_output_preserves_column(output: &ProjectionOutput, column: &str) -> bool {
    output.field.name() == column && expr_is_identity_column(&output.expr, column)
}

fn expr_is_identity_column(expr: &Expr, column: &str) -> bool {
    match expr {
        Expr::Column(expr_column) => expr_column.name() == column,
        Expr::Alias { expr, alias } => alias == column && expr_is_identity_column(expr, column),
        _ => false,
    }
}

pub(super) fn join_output_schema(
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    outputs: &[JoinOutputColumn],
) -> Arc<Schema> {
    Arc::new(Schema::new(
        outputs
            .iter()
            .map(|output| {
                let input_schema = match output.side {
                    JoinSide::Left => left_schema,
                    JoinSide::Right => right_schema,
                };
                let input_field = field_with_name(input_schema, output.column.name())
                    .expect("join output column must exist on input schema");
                Field::new(output.output_name.clone(), input_field.data_type().clone(), input_field.is_nullable())
            })
            .collect::<Vec<_>>(),
    ))
}
