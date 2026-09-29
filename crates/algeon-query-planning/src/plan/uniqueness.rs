//! Uniqueness keys derived from plan structure and source facts.
//!
//! A key is a set of output columns on which a node's rows are distinct.
//! [`partial_plan_node_is_unique_on`] answers for a plan still being built.

use std::collections::{BTreeMap, BTreeSet};

use arrow::datatypes::SchemaRef;

use super::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryNode, WindowFunction};
use crate::{
    expr::{BinaryOp, Expr, LiteralValue, alias_wrapped_column, representation_cast_preserves_identity},
    source::SourceNode,
};

pub(crate) fn partial_plan_node_is_unique_on(nodes: &[QueryNode], node_id: NodeId, columns: &BTreeSet<String>) -> bool {
    unique_keys_by_node(nodes).get(node_id.index()).is_some_and(|keys| keys.iter().any(|key| key.is_subset(columns)))
}

fn unique_keys_by_node(nodes: &[QueryNode]) -> Vec<Vec<BTreeSet<String>>> {
    let mut by_node = Vec::with_capacity(nodes.len());
    for node in nodes {
        let input_keys = |input: NodeId| by_node.get(input.index()).cloned().unwrap_or_default();
        let keys = match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.unique_key_columns.clone(),
                SourceNode::GpuResident(source) => source.unique_key_columns.iter().cloned().collect(),
                SourceNode::LiteralBatch(_) => Vec::new(),
            },
            PlanNode::Filter(filter) => input_keys(filter.input),
            PlanNode::Sort(sort) => input_keys(sort.input),
            PlanNode::Limit(limit) => input_keys(limit.input),
            PlanNode::Projection(projection) => {
                let input_schema = nodes.get(projection.input.index()).map(|node| node.output_schema());
                let mappings = projection.outputs.iter().filter_map(|output| {
                    Some((unique_source_column(&output.expr, input_schema)?.to_owned(), output.field.name().clone()))
                });
                remap_unique_key_sets(&input_keys(projection.input), mappings)
            },
            PlanNode::Aggregate(aggregate)
                if aggregate.spec.grouping_sets.is_none() && !aggregate.spec.group_exprs.is_empty() =>
            {
                vec![
                    node.output_schema()
                        .fields()
                        .iter()
                        .take(aggregate.spec.group_exprs.len())
                        .map(|field| field.name().clone())
                        .collect(),
                ]
            },
            PlanNode::Aggregate(_) => Vec::new(),
            PlanNode::Window(window) => {
                let mut keys = input_keys(window.input);
                if window.spec.partition_by.is_empty() {
                    keys.extend(window.spec.functions.iter().filter_map(|function| match function {
                        WindowFunction::RowNumber { output } => Some(BTreeSet::from([output.name().clone()])),
                        _ => None,
                    }));
                }
                keys
            },
            PlanNode::Join(join) => join_unique_keys(join, &input_keys(join.inputs[0]), &input_keys(join.inputs[1])),
            PlanNode::GraphAlgorithm(graph) => {
                let key = graph.spec.unique_output_key_columns();
                if !key.is_empty() && key.iter().all(|column| node.output_schema().field_with_name(column).is_ok()) {
                    vec![key.iter().map(|column| (*column).to_owned()).collect()]
                } else {
                    Vec::new()
                }
            },
            PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::VectorAlgorithm(_)
            | PlanNode::MlAlgorithm(_) => Vec::new(),
        };
        by_node.push(keys);
    }
    by_node
}

fn join_unique_keys(
    join: &JoinNode,
    left_keys: &[BTreeSet<String>],
    right_keys: &[BTreeSet<String>],
) -> Vec<BTreeSet<String>> {
    let side_is_unique = |side: JoinSide, keys: &[BTreeSet<String>]| {
        let columns = join
            .spec
            .keys
            .iter()
            .map(|key| match side {
                JoinSide::Left => key.left.name().to_owned(),
                JoinSide::Right => key.right.name().to_owned(),
            })
            .collect::<BTreeSet<_>>();
        keys.iter().any(|key| key.is_subset(&columns))
    };
    let bounded_side = [JoinSide::Left, JoinSide::Right]
        .into_iter()
        .find(|side| join.spec.has_at_most_one_match_per_bounded_row(*side));
    let left_is_unique = side_is_unique(JoinSide::Left, left_keys);
    let right_is_unique = side_is_unique(JoinSide::Right, right_keys);
    let bounded_is_unique = bounded_side.is_some_and(|side| match side {
        JoinSide::Left => left_is_unique,
        JoinSide::Right => right_is_unique,
    });
    let mappings = |side| {
        join.spec
            .output_projection
            .iter()
            .filter(move |column| column.side == side)
            .map(|column| (column.column.name().to_owned(), column.output_name.clone()))
    };
    let mut output = Vec::new();
    if bounded_side == Some(JoinSide::Left)
        || bounded_is_unique
        || right_is_unique
        || matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti { .. })
    {
        output.extend(remap_unique_key_sets(left_keys, mappings(JoinSide::Left)));
    }
    if bounded_side == Some(JoinSide::Right) || bounded_is_unique || left_is_unique {
        output.extend(remap_unique_key_sets(right_keys, mappings(JoinSide::Right)));
    }
    output
}

fn remap_unique_key_sets(
    input: &[BTreeSet<String>],
    mappings: impl IntoIterator<Item = (String, String)>,
) -> Vec<BTreeSet<String>> {
    let mappings = mappings.into_iter().fold(BTreeMap::<_, Vec<_>>::new(), |mut mappings, (input, output)| {
        mappings.entry(input).or_default().push(output);
        mappings
    });
    input
        .iter()
        .flat_map(|key| {
            key.iter()
                .try_fold(vec![BTreeSet::new()], |keys, column| {
                    let outputs = mappings.get(column)?;
                    Some(
                        keys.into_iter()
                            .flat_map(|key| {
                                outputs.iter().map(move |output| {
                                    let mut key = key.clone();
                                    key.insert(output.clone());
                                    key
                                })
                            })
                            .collect(),
                    )
                })
                .unwrap_or_default()
        })
        .collect()
}

fn identity_source_column<'a>(expr: &'a Expr, input_schema: Option<&SchemaRef>) -> Option<&'a str> {
    if let Some(column) = alias_wrapped_column(expr) {
        return Some(column.name());
    }
    match expr {
        Expr::Alias { expr, .. } => identity_source_column(expr, input_schema),
        Expr::Cast { expr, target_type } => {
            let column = identity_source_column(expr, input_schema)?;
            let source_type = input_schema?.field_with_name(column).ok()?.data_type();
            representation_cast_preserves_identity(source_type, target_type).then_some(column)
        },
        _ => None,
    }
}

fn unique_source_column<'a>(expr: &'a Expr, input_schema: Option<&SchemaRef>) -> Option<&'a str> {
    if let Some(column) = identity_source_column(expr, input_schema) {
        return Some(column);
    }
    let Expr::Binary { left, op: BinaryOp::Add, right, .. } = expr else {
        return None;
    };
    let (column, literal) = match (left.as_ref(), right.as_ref()) {
        (Expr::Column(column), Expr::Literal(literal)) | (Expr::Literal(literal), Expr::Column(column)) => {
            (column, literal)
        },
        _ => return None,
    };
    let field = input_schema?.field_with_name(column.name()).ok()?;
    let source_type = field.data_type();
    (!field.is_nullable()
        && source_type.is_integer()
        && literal.data_type() == *source_type
        && matches!(
            literal,
            LiteralValue::Int8(0)
                | LiteralValue::Int16(0)
                | LiteralValue::Int32(0)
                | LiteralValue::Int64(0)
                | LiteralValue::UInt8(0)
                | LiteralValue::UInt16(0)
                | LiteralValue::UInt32(0)
                | LiteralValue::UInt64(0)
        ))
    .then_some(column.name())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateSpec, ColumnRef, SortExpr},
        plan::{JoinKey, JoinOutputColumn, JoinSpec, QueryPlanBuilder, WindowSpec},
        source::LocalParquetSource,
    };

    fn endpoint_source() -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sourceId", DataType::Utf8, false),
            Field::new("targetId", DataType::Utf8, false),
        ]));
        LocalParquetSource::new("/tmp/endpoints.parquet", schema, vec!["sourceId".to_owned(), "targetId".to_owned()])
    }

    #[test]
    fn builder_answers_partial_uniqueness_without_finishing_a_temporary_plan() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(endpoint_source()));
        let aggregate = builder
            .add_aggregate(
                source,
                AggregateSpec::new(vec![Expr::column("sourceId")], Vec::new()),
                Arc::new(Schema::new(vec![Field::new("sourceId", DataType::Utf8, false)])),
            )
            .unwrap();

        assert!(builder.output_is_unique_on(aggregate, &BTreeSet::from(["sourceId".to_owned()])));
        assert!(!builder.output_is_unique_on(source, &BTreeSet::from(["targetId".to_owned()])));
    }

    #[test]
    fn integer_addition_requires_non_nullable_identity_without_overflow() {
        let non_nullable = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let nullable = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));
        let add =
            |literal| Expr::binary(Expr::column("value"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(literal)));
        assert_eq!(unique_source_column(&add(0), Some(&non_nullable)), Some("value"));
        assert_eq!(unique_source_column(&add(1), Some(&non_nullable)), None);
        assert_eq!(unique_source_column(&add(0), Some(&nullable)), None);
    }

    #[test]
    fn row_number_over_a_grouped_relation_is_unique_on_its_dense_id() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(endpoint_source()));
        let codebook = builder
            .add_aggregate(
                source,
                AggregateSpec::new(vec![Expr::column("sourceId")], Vec::new()),
                Arc::new(Schema::new(vec![Field::new("sourceId", DataType::Utf8, false)])),
            )
            .unwrap();
        let dense = builder
            .add_window(
                codebook,
                WindowSpec::new(
                    Vec::new(),
                    vec![SortExpr::new(Expr::column("sourceId"), true, false)],
                    vec![WindowFunction::row_number_i64("__dense_id")],
                ),
            )
            .unwrap();

        assert!(builder.output_is_unique_on(dense, &BTreeSet::from(["__dense_id".to_owned()])));
    }

    #[test]
    fn left_semi_and_anti_joins_keep_the_left_unique_key() {
        let key_schema = Arc::new(Schema::new(vec![
            Field::new("sourceId", DataType::Utf8, false),
            Field::new("targetId", DataType::Utf8, false),
        ]));
        for join_type in [JoinType::LeftSemi, JoinType::LEFT_ANTI] {
            let mut builder = QueryPlanBuilder::new();
            let relation = builder.add_source(SourceNode::LocalParquet(endpoint_source()));
            let unique_relation = builder
                .add_aggregate(
                    relation,
                    AggregateSpec::new(vec![Expr::column("sourceId"), Expr::column("targetId")], Vec::new()),
                    Arc::clone(&key_schema),
                )
                .unwrap();
            let lookup = builder.add_source(SourceNode::LocalParquet(endpoint_source()));
            let selected = builder
                .add_join(
                    unique_relation,
                    lookup,
                    JoinSpec::new(
                        join_type,
                        vec![JoinKey::new(
                            ColumnRef::input(0, "sourceId", "sourceId"),
                            ColumnRef::input(1, "sourceId", "sourceId"),
                        )],
                    )
                    .with_output_projection(vec![
                        JoinOutputColumn::left(ColumnRef::input(0, "sourceId", "sourceId"), "sourceId"),
                        JoinOutputColumn::left(ColumnRef::input(0, "targetId", "targetId"), "targetId"),
                    ]),
                    Arc::clone(&key_schema),
                )
                .unwrap();

            assert!(
                builder.output_is_unique_on(selected, &BTreeSet::from(["sourceId".to_owned(), "targetId".to_owned()]))
            );
        }
    }
}
