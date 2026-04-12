use super::*;

pub(super) fn seed_reader_predicates_from_group_key_filters(plan: QueryPlan) -> (QueryPlan, bool) {
    let mut nodes = plan.nodes().to_vec();
    let parent_counts = rewrite::parent_counts(&plan);
    let mut changed = false;

    for node in plan.nodes() {
        let PlanNode::Filter(filter) = &node.kind else {
            continue;
        };
        for conjunct in predicate_conjuncts(&filter.predicate) {
            changed |= push_filter_conjunct_to_reader_source(
                &plan,
                &parent_counts,
                &mut nodes,
                filter.input,
                conjunct.clone(),
                false,
                false,
            );
        }
    }

    if !changed {
        return (plan, false);
    }
    let root = plan.root_id();
    (rewrite::finish_rewritten_plan(nodes, root, plan.sink()).with_rewrite_state_from(&plan), true)
}

/// Returns a copy of `plan` with eligible post-aggregate group-key filters
/// mirrored onto local Parquet dimension-reader predicates.
///
/// The original post-aggregate filter remains in the plan. This helper is used
/// by adapter-side object-store pruning, where row-group statistics are not
/// available inside `nexus-query-engine` but the seeded dimension predicate is
/// still needed to derive a key range.
pub fn seed_dimension_derived_reader_predicates(plan: QueryPlan) -> QueryPlan {
    seed_reader_predicates_from_group_key_filters(plan).0
}

fn predicate_conjuncts(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::Binary { left, op: BinaryOp::And, right, .. } => {
            let mut conjuncts = predicate_conjuncts(left);
            conjuncts.extend(predicate_conjuncts(right));
            conjuncts
        },
        _ => vec![expr],
    }
}

fn push_filter_conjunct_to_reader_source(
    plan: &QueryPlan,
    parent_counts: &[usize],
    nodes: &mut [QueryNode],
    node_id: NodeId,
    predicate: Expr,
    crossed_aggregate: bool,
    crossed_shared_path: bool,
) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    let crossed_shared_path =
        crossed_shared_path || parent_counts.get(node_id.index()).copied().unwrap_or_default() > 1;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source)
                if crossed_aggregate && !crossed_shared_path && seed_predicate_is_supported(source, &predicate) =>
            {
                let Some(source) = mutable_local_parquet_source(nodes, node_id) else {
                    return false;
                };
                if source.reader_predicate.as_ref() == Some(&predicate) {
                    return false;
                }
                source.reader_predicate = Some(match source.reader_predicate.take() {
                    Some(existing) => Expr::binary(existing, BinaryOp::And, predicate),
                    None => predicate,
                });
                source.refresh_diagnostics();
                true
            },
            _ => false,
        },
        PlanNode::Filter(filter) => push_filter_conjunct_to_reader_source(
            plan,
            parent_counts,
            nodes,
            filter.input,
            predicate,
            crossed_aggregate,
            crossed_shared_path,
        ),
        PlanNode::Projection(projection) => {
            remap_projection_predicate(&predicate, projection).is_some_and(|predicate| {
                push_filter_conjunct_to_reader_source(
                    plan,
                    parent_counts,
                    nodes,
                    projection.input,
                    predicate,
                    crossed_aggregate,
                    crossed_shared_path,
                )
            })
        },
        PlanNode::Aggregate(aggregate) => {
            remap_aggregate_group_predicate(&predicate, aggregate).is_some_and(|predicate| {
                push_filter_conjunct_to_reader_source(
                    plan,
                    parent_counts,
                    nodes,
                    aggregate.input,
                    predicate,
                    true,
                    crossed_shared_path,
                )
            })
        },
        PlanNode::Join(join) => remap_join_output_predicate(&predicate, join).is_some_and(|(side, predicate)| {
            let input = match side {
                JoinSide::Left => join.inputs[0],
                JoinSide::Right => join.inputs[1],
            };
            push_filter_conjunct_to_reader_source(
                plan,
                parent_counts,
                nodes,
                input,
                predicate,
                crossed_aggregate,
                crossed_shared_path,
            )
        }),
        PlanNode::Sort(sort) => push_filter_conjunct_to_reader_source(
            plan,
            parent_counts,
            nodes,
            sort.input,
            predicate,
            crossed_aggregate,
            crossed_shared_path,
        ),
        PlanNode::Limit(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
    }
}

fn seed_predicate_is_supported(source: &LocalParquetSource, predicate: &Expr) -> bool {
    if !predicate_is_reader_visible(source, predicate) {
        return false;
    }
    let combined = match source.reader_predicate.as_ref() {
        Some(existing) => Expr::binary(existing.clone(), BinaryOp::And, predicate.clone()),
        None => predicate.clone(),
    };
    source.reader_predicate_is_supported(&combined)
}

fn predicate_is_reader_visible(source: &LocalParquetSource, predicate: &Expr) -> bool {
    let columns = predicate_columns(predicate);
    !columns.is_empty()
        && columns.iter().all(|column| matches!(source.source_column_ref(column), Some(SourceColumnRef::File(_))))
}

fn remap_projection_predicate(predicate: &Expr, projection: &crate::plan::ProjectionNode) -> Option<Expr> {
    remap_unqualified_columns(predicate, &mut |column| {
        let output = projection.outputs.iter().find(|output| output.field.name() == column.name())?;
        simple_projection_expr(&output.expr)
    })
}

fn remap_aggregate_group_predicate(predicate: &Expr, aggregate: &crate::plan::AggregateNode) -> Option<Expr> {
    if aggregate.spec.grouping_sets.is_some() {
        return None;
    }
    remap_unqualified_columns(predicate, &mut |column| {
        aggregate
            .spec
            .group_exprs
            .iter()
            .find(|expr| aggregate_group_output_name(expr) == Some(column.name()))
            .and_then(simple_projection_expr)
    })
}

fn remap_join_output_predicate(predicate: &Expr, join: &JoinNode) -> Option<(JoinSide, Expr)> {
    if !matches!(join.spec.join_type, JoinType::Inner | JoinType::Left) || join.spec.output_projection.is_empty() {
        return None;
    }
    let mut side = None;
    let remapped = remap_unqualified_columns(predicate, &mut |column| {
        let output = join.spec.output_projection.iter().find(|output| output.output_name == column.name())?;
        if join.spec.join_type == JoinType::Left && output.side != JoinSide::Left {
            return None;
        }
        match side {
            Some(existing) if existing != output.side => return None,
            Some(_) => {},
            None => side = Some(output.side),
        }
        Some(Expr::column(output.column.name()))
    })?;
    side.map(|side| (side, remapped))
}

// Deliberately narrower than the similarly named helper in
// `join_filter_reassociation`: dimension pruning must reject expression
// families it cannot prove safe to push into source pruning.
pub(super) fn remap_unqualified_columns(
    expr: &Expr,
    map_column: &mut impl FnMut(&ColumnRef) -> Option<Expr>,
) -> Option<Expr> {
    match expr {
        Expr::Column(column) if column.qualifier().is_none() => map_column(column),
        Expr::Column(_) => None,
        Expr::Literal(value) => Some(Expr::Literal(value.clone())),
        Expr::Alias { expr, alias } => Some(remap_unqualified_columns(expr, map_column)?.alias(alias.clone())),
        Expr::Cast { expr, target_type } => {
            Some(Expr::cast(remap_unqualified_columns(expr, map_column)?, target_type.clone()))
        },
        Expr::TryCast { expr, target_type } => {
            Some(Expr::try_cast(remap_unqualified_columns(expr, map_column)?, target_type.clone()))
        },
        Expr::Binary { left, op, right, output_type } => Some(Expr::Binary {
            left: Box::new(remap_unqualified_columns(left, map_column)?),
            op: *op,
            right: Box::new(remap_unqualified_columns(right, map_column)?),
            output_type: output_type.clone(),
        }),
        Expr::Not(expr) => Some(Expr::logical_not(remap_unqualified_columns(expr, map_column)?)),
        Expr::IsNull(expr) => Some(Expr::is_null(remap_unqualified_columns(expr, map_column)?)),
        Expr::IsNotNull(expr) => Some(Expr::is_not_null(remap_unqualified_columns(expr, map_column)?)),
        Expr::InList { expr, list, negated } => Some(Expr::InList {
            expr: Box::new(remap_unqualified_columns(expr, map_column)?),
            list: list.iter().map(|expr| remap_unqualified_columns(expr, map_column)).collect::<Option<Vec<_>>>()?,
            negated: *negated,
        }),
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => None,
    }
}

fn simple_projection_expr(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Column(column) => Some(Expr::column(column.name())),
        Expr::Alias { expr, .. } => simple_projection_expr(expr),
        _ => None,
    }
}

pub(super) fn aggregate_group_output_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { alias, .. } => Some(alias.as_str()),
        _ => None,
    }
}
