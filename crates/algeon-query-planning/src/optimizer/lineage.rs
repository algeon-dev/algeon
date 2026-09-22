use crate::{
    expr::Expr,
    plan::{AggregateNode, JoinSide, NodeId, PlanNode, QueryPlan},
    source::SourceNode,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SourceColumnLineage {
    pub(super) source_node_id: NodeId,
    pub(super) source_column: String,
    pub(super) aggregate_group_key_node_id: Option<NodeId>,
    pub(super) has_shared_path: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AggregateGroupKeyLineage {
    Producer,
    Consumer,
    Reject,
}

pub(super) fn unique_lineage(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Option<SourceColumnLineage> {
    let mut lineages =
        dedup_lineages(lineages_for_column(plan, parent_counts, node_id, column_name, aggregate_group_key_lineage));
    (lineages.len() == 1).then(|| lineages.remove(0))
}

pub(super) fn dedup_lineages(mut lineages: Vec<SourceColumnLineage>) -> Vec<SourceColumnLineage> {
    lineages.sort_by(|left, right| {
        left.source_node_id
            .cmp(&right.source_node_id)
            .then_with(|| left.source_column.cmp(&right.source_column))
            .then_with(|| left.aggregate_group_key_node_id.cmp(&right.aggregate_group_key_node_id))
            .then_with(|| left.has_shared_path.cmp(&right.has_shared_path))
    });
    lineages.dedup();
    lineages
}

pub(super) fn lineages_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Vec<SourceColumnLineage> {
    let Some(node) = plan.node(node_id) else {
        return Vec::new();
    };
    let has_shared_path = parent_counts[node_id.index()] > 1;
    let mark_shared = |mut lineage: SourceColumnLineage| {
        lineage.has_shared_path |= has_shared_path;
        lineage
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.source_column_ref(column_name).is_some() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            SourceNode::GpuResident(source) if source.schema.field_with_name(column_name).is_ok() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            SourceNode::LiteralBatch(source) if source.schema.field_with_name(column_name).is_ok() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            _ => Vec::new(),
        },
        PlanNode::Filter(filter) => {
            lineages_for_column(plan, parent_counts, filter.input, column_name, aggregate_group_key_lineage)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .filter(|output| output.field.name() == column_name)
            .flat_map(|output| {
                projection_expr_lineages(
                    plan,
                    parent_counts,
                    projection.input,
                    &output.expr,
                    aggregate_group_key_lineage,
                )
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Sort(sort) => {
            lineages_for_column(plan, parent_counts, sort.input, column_name, aggregate_group_key_lineage)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Limit(_) if aggregate_group_key_lineage == AggregateGroupKeyLineage::Consumer => Vec::new(),
        PlanNode::Limit(limit) => {
            lineages_for_column(plan, parent_counts, limit.input, column_name, aggregate_group_key_lineage)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Join(join) => join
            .spec
            .output_projection
            .iter()
            .filter(|output| output.output_name == column_name)
            .flat_map(|output| {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                lineages_for_column(plan, parent_counts, input, output.column.name(), aggregate_group_key_lineage)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Aggregate(aggregate) if aggregate_group_key_lineage != AggregateGroupKeyLineage::Reject => {
            aggregate_group_key_lineages(
                plan,
                parent_counts,
                aggregate,
                node_id,
                column_name,
                aggregate_group_key_lineage,
            )
            .into_iter()
            .map(mark_shared)
            .collect()
        },
        PlanNode::Aggregate(_) => Vec::new(),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .flat_map(|input| {
                lineages_for_column(plan, parent_counts, *input, column_name, aggregate_group_key_lineage)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => Vec::new(),
    }
}

fn aggregate_group_key_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    aggregate: &AggregateNode,
    aggregate_node_id: NodeId,
    column_name: &str,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Vec<SourceColumnLineage> {
    if aggregate.spec.grouping_sets.is_some() {
        return Vec::new();
    }
    aggregate
        .spec
        .group_exprs
        .iter()
        .filter_map(|expr| aggregate_group_key_output_name(expr).map(|output_name| (expr, output_name)))
        .filter(|(_, output_name)| *output_name == column_name)
        .flat_map(|(expr, output_name)| match aggregate_group_key_lineage {
            AggregateGroupKeyLineage::Producer => {
                projection_expr_lineages(plan, parent_counts, aggregate.input, expr, aggregate_group_key_lineage)
                    .into_iter()
                    .map(move |mut lineage| {
                        lineage.source_column = output_name.to_string();
                        lineage
                    })
                    .collect::<Vec<_>>()
            },
            AggregateGroupKeyLineage::Consumer => {
                consumer_pre_aggregate_expr_lineages(plan, parent_counts, aggregate.input, expr)
            },
            AggregateGroupKeyLineage::Reject => Vec::new(),
        })
        .map(|mut lineage| {
            lineage.aggregate_group_key_node_id = Some(aggregate_node_id);
            lineage
        })
        .collect()
}

pub(super) fn aggregate_group_key_output_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { alias, .. } => Some(alias.as_str()),
        _ => None,
    }
}

pub(super) fn producer_input_is_direct_aggregate_keyset_alias(
    plan: &QueryPlan,
    node_id: NodeId,
    column_name: &str,
    aggregate_node_id: NodeId,
) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    if node_id == aggregate_node_id {
        return matches!(node.kind, PlanNode::Aggregate(_))
            && node.output_schema().field_with_name(column_name).is_ok();
    }
    match &node.kind {
        PlanNode::Projection(projection) => {
            projection.outputs.iter().filter(|output| output.field.name() == column_name).any(|output| {
                producer_expr_is_direct_aggregate_keyset_alias(plan, projection.input, &output.expr, aggregate_node_id)
            })
        },
        PlanNode::Sort(sort) => {
            producer_input_is_direct_aggregate_keyset_alias(plan, sort.input, column_name, aggregate_node_id)
        },
        PlanNode::Source(_)
        | PlanNode::Filter(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Limit(_)
        | PlanNode::Join(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => false,
    }
}

fn producer_expr_is_direct_aggregate_keyset_alias(
    plan: &QueryPlan,
    input: NodeId,
    expr: &Expr,
    aggregate_node_id: NodeId,
) -> bool {
    match expr {
        Expr::Column(column) => {
            producer_input_is_direct_aggregate_keyset_alias(plan, input, column.name(), aggregate_node_id)
        },
        Expr::Alias { expr, .. } => {
            producer_expr_is_direct_aggregate_keyset_alias(plan, input, expr, aggregate_node_id)
        },
        _ => false,
    }
}

fn projection_expr_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    input: NodeId,
    expr: &Expr,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Vec<SourceColumnLineage> {
    match expr {
        Expr::Column(column) => {
            lineages_for_column(plan, parent_counts, input, column.name(), aggregate_group_key_lineage)
        },
        Expr::Alias { expr, .. } => {
            projection_expr_lineages(plan, parent_counts, input, expr, aggregate_group_key_lineage)
        },
        _ => Vec::new(),
    }
}

fn consumer_pre_aggregate_expr_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    input: NodeId,
    expr: &Expr,
) -> Vec<SourceColumnLineage> {
    match expr {
        Expr::Column(column) => consumer_pre_aggregate_lineages_for_column(plan, parent_counts, input, column.name()),
        Expr::Alias { expr, .. } => consumer_pre_aggregate_expr_lineages(plan, parent_counts, input, expr),
        _ => Vec::new(),
    }
}

fn consumer_pre_aggregate_lineages_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
) -> Vec<SourceColumnLineage> {
    let Some(node) = plan.node(node_id) else {
        return Vec::new();
    };
    let has_shared_path = parent_counts[node_id.index()] > 1;
    let mark_shared = |mut lineage: SourceColumnLineage| {
        lineage.has_shared_path |= has_shared_path;
        lineage
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.source_column_ref(column_name).is_some() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            _ => Vec::new(),
        },
        PlanNode::Filter(filter) => {
            consumer_pre_aggregate_lineages_for_column(plan, parent_counts, filter.input, column_name)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .filter(|output| output.field.name() == column_name)
            .flat_map(|output| {
                consumer_pre_aggregate_expr_lineages(plan, parent_counts, projection.input, &output.expr)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Join(join) => join
            .spec
            .output_projection
            .iter()
            .filter(|output| output.output_name == column_name)
            .flat_map(|output| {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                consumer_pre_aggregate_lineages_for_column(plan, parent_counts, input, output.column.name())
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .flat_map(|input| consumer_pre_aggregate_lineages_for_column(plan, parent_counts, *input, column_name))
            .map(mark_shared)
            .collect(),
        PlanNode::Aggregate(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => Vec::new(),
    }
}

pub(super) fn unique_simple_lineage(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
) -> Option<SourceColumnLineage> {
    let mut lineages = dedup_lineages(simple_lineages_for_column(plan, parent_counts, node_id, column_name));
    (lineages.len() == 1).then(|| lineages.remove(0))
}

pub(super) fn simple_lineages_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
) -> Vec<SourceColumnLineage> {
    let Some(node) = plan.node(node_id) else {
        return Vec::new();
    };
    let has_shared_path = parent_counts[node_id.index()] > 1;
    let mark_shared = |mut lineage: SourceColumnLineage| {
        lineage.has_shared_path |= has_shared_path;
        lineage
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.source_column_ref(column_name).is_some() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            _ => Vec::new(),
        },
        PlanNode::Filter(filter) => simple_lineages_for_column(plan, parent_counts, filter.input, column_name)
            .into_iter()
            .map(mark_shared)
            .collect(),
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .filter(|output| output.field.name() == column_name)
            .flat_map(|output| simple_projection_expr_lineages(plan, parent_counts, projection.input, &output.expr))
            .map(mark_shared)
            .collect(),
        PlanNode::Sort(sort) => simple_lineages_for_column(plan, parent_counts, sort.input, column_name)
            .into_iter()
            .map(mark_shared)
            .collect(),
        PlanNode::Limit(_) => Vec::new(),
        PlanNode::Join(join) => join
            .spec
            .output_projection
            .iter()
            .filter(|output| output.output_name == column_name)
            .flat_map(|output| {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                simple_lineages_for_column(plan, parent_counts, input, output.column.name())
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .flat_map(|input| simple_lineages_for_column(plan, parent_counts, *input, column_name))
            .map(mark_shared)
            .collect(),
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_) => Vec::new(),
    }
}

fn simple_projection_expr_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    input: NodeId,
    expr: &Expr,
) -> Vec<SourceColumnLineage> {
    match expr {
        Expr::Column(column) => simple_lineages_for_column(plan, parent_counts, input, column.name()),
        Expr::Alias { expr, .. } => simple_projection_expr_lineages(plan, parent_counts, input, expr),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        optimizer::rewrite::parent_counts,
        plan::{
            QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmParams, VectorAlgorithmSpec, VectorPcaParams,
            VectorRelationBinding, VectorRowIdBinding, vector_pca_output_schema,
        },
        source::{LocalParquetSource, SourceNode},
    };

    #[test]
    fn vector_algorithm_is_a_source_column_lineage_barrier() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/vector-lineage.parquet",
            schema,
            vec!["id".to_string(), "x".to_string()],
        )));
        let params =
            VectorPcaParams::new(VectorRelationBinding::new(source, VectorRowIdBinding::column("id"), ["x"]), 1);
        let vector = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
            VectorAlgorithm::Pca,
            VectorAlgorithmParams::Pca(params),
            vector_pca_output_schema(DataType::Int64, 1),
        ));
        let plan = builder.finish_gpu_resident(vector);
        let parents = parent_counts(&plan);

        assert_eq!(simple_lineages_for_column(&plan, &parents, source, "x").len(), 1);
        assert!(simple_lineages_for_column(&plan, &parents, vector, "component_0").is_empty());
        assert!(simple_lineages_for_column(&plan, &parents, vector, "row_id").is_empty());
    }
}
