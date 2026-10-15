//! Source-column lineage for optimizer rules.
//!
//! A lineage says where a column visible at a node came from: a source node
//! and column, whether the path crossed an aggregate group key, and whether any
//! node on the path has more than one consumer. Dynamic-filter route discovery
//! uses a unique lineage as the proof that a keyset or predicate stated on the
//! node's column can be stated on the source column. The walks differ in what
//! they cross: [`lineages_for_column`] crosses sorts, limits and aggregate
//! group keys under an [`AggregateGroupKeyLineage`] policy;
//! [`simple_lineages_for_column`] stops at aggregates and limits; the walk
//! below a consumed aggregate crosses only filters, projections, joins and
//! unions over local Parquet sources, because the keyset it proves is applied
//! by a Parquet reader.

use crate::{
    expr::{CaseWhen, Expr},
    plan::{AggregateNode, JoinSide, NodeId, PlanNode, QueryPlan},
    source::SourceNode,
};

/// One origin of a column: a source column, possibly seen through an aggregate
/// group key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SourceColumnLineage {
    pub(super) source_node_id: NodeId,
    pub(super) source_column: String,
    /// Aggregate whose group key the path crossed. The column's value set is
    /// then that aggregate's key set, which has one row per distinct key.
    pub(super) aggregate_group_key_node_id: Option<NodeId>,
    /// Whether any node between the queried node and the source has more than
    /// one consumer. Narrowing the source would then narrow those consumers.
    pub(super) has_shared_path: bool,
}

/// What an aggregate on the path means for the lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AggregateGroupKeyLineage {
    /// The caller produces a keyset at the aggregate: a group key stands for
    /// the source column under the aggregate's output name.
    Producer,
    /// The caller applies a keyset below the aggregate at a Parquet reader:
    /// the group key is followed to its pre-aggregate Parquet column, and a
    /// limit above the aggregate ends the lineage because the reader sees
    /// rows the limit has not yet cut.
    Consumer,
    /// An aggregate ends the lineage.
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalParquetSortLineage {
    Stop,
    Cross,
}

/// The one lineage of `column_name` at `node_id`, or `None` when the column has
/// no traceable origin or more than one, as below a union of two sources.
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
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => Vec::new(),
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
            AggregateGroupKeyLineage::Consumer => local_parquet_lineages_for_expr(
                plan,
                parent_counts,
                aggregate.input,
                expr,
                LocalParquetSortLineage::Stop,
            ),
            AggregateGroupKeyLineage::Reject => Vec::new(),
        })
        .map(|mut lineage| {
            lineage.aggregate_group_key_node_id = Some(aggregate_node_id);
            lineage
        })
        .collect()
}

/// Output name of a group key that is a column or an aliased expression; a
/// computed key without an alias has no name a consumer can refer to.
pub(super) fn aggregate_group_key_output_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { alias, .. } => Some(alias.as_str()),
        _ => None,
    }
}

/// Whether `column_name` at `node_id` is an output of `aggregate_node_id`
/// reached only through renaming projections and sorts, so its value set is
/// the keyset that aggregate produces, with nothing added or removed.
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
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
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

fn local_parquet_lineages_for_expr(
    plan: &QueryPlan,
    parent_counts: &[usize],
    input: NodeId,
    expr: &Expr,
    sort_lineage: LocalParquetSortLineage,
) -> Vec<SourceColumnLineage> {
    match expr {
        Expr::Column(column) => {
            local_parquet_lineages_for_column(plan, parent_counts, input, column.name(), sort_lineage)
        },
        Expr::Alias { expr, .. } => local_parquet_lineages_for_expr(plan, parent_counts, input, expr, sort_lineage),
        _ => Vec::new(),
    }
}

fn local_parquet_lineages_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
    sort_lineage: LocalParquetSortLineage,
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
            local_parquet_lineages_for_column(plan, parent_counts, filter.input, column_name, sort_lineage)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .filter(|output| output.field.name() == column_name)
            .flat_map(|output| {
                local_parquet_lineages_for_expr(plan, parent_counts, projection.input, &output.expr, sort_lineage)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Sort(sort) => match sort_lineage {
            LocalParquetSortLineage::Stop => Vec::new(),
            LocalParquetSortLineage::Cross => {
                local_parquet_lineages_for_column(plan, parent_counts, sort.input, column_name, sort_lineage)
                    .into_iter()
                    .map(mark_shared)
                    .collect()
            },
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
                local_parquet_lineages_for_column(plan, parent_counts, input, output.column.name(), sort_lineage)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .flat_map(|input| local_parquet_lineages_for_column(plan, parent_counts, *input, column_name, sort_lineage))
            .map(mark_shared)
            .collect(),
        PlanNode::Aggregate(_)
        | PlanNode::Limit(_)
        | PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => Vec::new(),
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
    local_parquet_lineages_for_column(plan, parent_counts, node_id, column_name, LocalParquetSortLineage::Cross)
}

/// Replaces unqualified column references while preserving expression structure.
pub(super) fn substitute_unqualified_columns(expr: Expr, map: &dyn Fn(&str) -> Option<Expr>) -> Expr {
    match expr {
        Expr::Column(column) => {
            if column.qualifier().is_none()
                && let Some(replacement) = map(column.name())
            {
                return replacement;
            }
            Expr::Column(column)
        },
        Expr::Literal(_) => expr,
        Expr::Not(inner) => Expr::Not(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::IsNull(inner) => Expr::IsNull(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::IsNotNull(inner) => Expr::IsNotNull(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::Cast { expr, target_type } => {
            Expr::Cast { expr: Box::new(substitute_unqualified_columns(*expr, map)), target_type }
        },
        Expr::TryCast { expr, target_type } => {
            Expr::TryCast { expr: Box::new(substitute_unqualified_columns(*expr, map)), target_type }
        },
        Expr::Binary { left, op, right, output_type } => Expr::Binary {
            left: Box::new(substitute_unqualified_columns(*left, map)),
            op,
            right: Box::new(substitute_unqualified_columns(*right, map)),
            output_type,
        },
        Expr::Like { expr, pattern, escape, negated } => {
            Expr::Like { expr: Box::new(substitute_unqualified_columns(*expr, map)), pattern, escape, negated }
        },
        Expr::Substring { expr, start, length } => {
            Expr::Substring { expr: Box::new(substitute_unqualified_columns(*expr, map)), start, length }
        },
        Expr::Lower(inner) => Expr::Lower(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::Upper(inner) => Expr::Upper(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::Trim { expr, trim_chars } => Expr::trim(substitute_unqualified_columns(*expr, map), trim_chars),
        Expr::LTrim { expr, trim_chars } => Expr::ltrim(substitute_unqualified_columns(*expr, map), trim_chars),
        Expr::RTrim { expr, trim_chars } => Expr::rtrim(substitute_unqualified_columns(*expr, map), trim_chars),
        Expr::StringContains { expr, pattern } => {
            Expr::string_contains(substitute_unqualified_columns(*expr, map), pattern)
        },
        Expr::StartsWith { expr, prefix } => Expr::starts_with(substitute_unqualified_columns(*expr, map), prefix),
        Expr::EndsWith { expr, suffix } => Expr::ends_with(substitute_unqualified_columns(*expr, map), suffix),
        Expr::StringLength { expr, unit, output_type } => {
            Expr::StringLength { expr: Box::new(substitute_unqualified_columns(*expr, map)), unit, output_type }
        },
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => Expr::RegexpReplace {
            expr: Box::new(substitute_unqualified_columns(*expr, map)),
            pattern,
            replacement,
            output_type,
        },
        Expr::Concat(parts) => {
            Expr::Concat(parts.into_iter().map(|part| substitute_unqualified_columns(part, map)).collect())
        },
        Expr::Abs(inner) => Expr::Abs(Box::new(substitute_unqualified_columns(*inner, map))),
        Expr::UnaryMath { op, expr, output_type } => {
            Expr::UnaryMath { op, expr: Box::new(substitute_unqualified_columns(*expr, map)), output_type }
        },
        Expr::Round { expr, decimal_places, output_type } => {
            Expr::Round { expr: Box::new(substitute_unqualified_columns(*expr, map)), decimal_places, output_type }
        },
        Expr::InList { expr, list, negated } => Expr::InList {
            expr: Box::new(substitute_unqualified_columns(*expr, map)),
            list: list.into_iter().map(|item| substitute_unqualified_columns(item, map)).collect(),
            negated,
        },
        Expr::IsDistinctFrom { left, right } => Expr::IsDistinctFrom {
            left: Box::new(substitute_unqualified_columns(*left, map)),
            right: Box::new(substitute_unqualified_columns(*right, map)),
        },
        Expr::IsNotDistinctFrom { left, right } => Expr::IsNotDistinctFrom {
            left: Box::new(substitute_unqualified_columns(*left, map)),
            right: Box::new(substitute_unqualified_columns(*right, map)),
        },
        Expr::Case { branches, else_expr } => Expr::Case {
            branches: branches
                .into_iter()
                .map(|branch| CaseWhen {
                    when: substitute_unqualified_columns(branch.when, map),
                    then: substitute_unqualified_columns(branch.then, map),
                })
                .collect(),
            else_expr: Box::new(substitute_unqualified_columns(*else_expr, map)),
        },
        Expr::Extract { field, expr } => {
            Expr::Extract { field, expr: Box::new(substitute_unqualified_columns(*expr, map)) }
        },
        Expr::DateTrunc { field, expr, output_type } => {
            Expr::DateTrunc { field, expr: Box::new(substitute_unqualified_columns(*expr, map)), output_type }
        },
        Expr::Alias { expr, alias } => substitute_unqualified_columns(*expr, map).alias(alias),
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
        let params = VectorPcaParams::new(VectorRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x"]), 1);
        let vector = builder
            .add_vector_algorithm(
                vec![source],
                VectorAlgorithmSpec::new(
                    VectorAlgorithm::Pca,
                    VectorAlgorithmParams::Pca(params),
                    vector_pca_output_schema(DataType::Int64, 1),
                ),
            )
            .unwrap();
        let plan = builder.finish_gpu_resident(vector).unwrap();
        let parents = parent_counts(&plan);

        assert_eq!(simple_lineages_for_column(&plan, &parents, source, "x").len(), 1);
        assert!(simple_lineages_for_column(&plan, &parents, vector, "component_0").is_empty());
        assert!(simple_lineages_for_column(&plan, &parents, vector, "row_id").is_empty());
    }
}
