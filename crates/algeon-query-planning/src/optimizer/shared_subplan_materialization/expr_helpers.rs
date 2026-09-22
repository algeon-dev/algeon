fn scalar_filtered_aggregate_supported(spec: &AggregateSpec) -> bool {
    spec.group_exprs.is_empty()
        && spec.grouping_sets.is_none()
        && !spec.aggr_exprs.is_empty()
        && spec.aggr_exprs.iter().all(scalar_filtered_aggregate_expr_supported)
}

fn scalar_filtered_aggregate_expr_supported(expr: &AggregateExpr) -> bool {
    if expr.filter.is_some() {
        return false;
    }
    if expr.distinct && expr.function != AggregateFunction::Count {
        return false;
    }
    match expr.function {
        AggregateFunction::Count if !expr.distinct && scalar_count_star_expr(&expr.expr) => true,
        AggregateFunction::Count
        | AggregateFunction::NUnique
        | AggregateFunction::Sum
        | AggregateFunction::Min
        | AggregateFunction::Max
        | AggregateFunction::Avg => scalar_aggregate_value_expr_supported(&expr.expr),
        AggregateFunction::Median => false,
        AggregateFunction::StddevSamp
        | AggregateFunction::StddevPop
        | AggregateFunction::VarSamp
        | AggregateFunction::VarPop => false,
    }
}

fn scalar_filtered_aggregate_streaming_supported(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().all(|expr| {
        !expr.distinct
            && matches!(
                expr.function,
                AggregateFunction::Count
                    | AggregateFunction::Sum
                    | AggregateFunction::Min
                    | AggregateFunction::Max
                    | AggregateFunction::Avg
            )
    })
}

fn scalar_count_star_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value) if !matches!(value, LiteralValue::Null(_)))
        || matches!(expr, Expr::Alias { expr, .. } if scalar_count_star_expr(expr))
}

fn scalar_count_star_aggregate_expr(plan: &QueryPlan, input: NodeId, expr: &Expr) -> bool {
    if scalar_count_star_expr(expr) {
        return true;
    }
    match expr {
        Expr::Column(column) => projected_count_preserving_column(plan, input, column.name()),
        Expr::Alias { expr, .. } => scalar_count_star_aggregate_expr(plan, input, expr),
        _ => false,
    }
}

fn projected_count_preserving_column(plan: &QueryPlan, node_id: NodeId, column: &str) -> bool {
    const HIDDEN_JOIN_ROW_PRESENCE_COLUMN: &str = "__algeon_native_join_row_presence";
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    match &node.kind {
        PlanNode::Projection(projection) => {
            projection.outputs.iter().find(|output| output.field.name() == column).is_some_and(|output| {
                match &output.expr {
                    expr if scalar_count_star_expr(expr) => true,
                    Expr::Column(inner) => projected_count_preserving_column(plan, projection.input, inner.name()),
                    Expr::Alias { expr, .. } => match expr.as_ref() {
                        Expr::Column(inner) => projected_count_preserving_column(plan, projection.input, inner.name()),
                        other => scalar_count_star_expr(other),
                    },
                    _ => false,
                }
            })
        },
        PlanNode::Join(_) => column == HIDDEN_JOIN_ROW_PRESENCE_COLUMN,
        PlanNode::Filter(filter) => projected_count_preserving_column(plan, filter.input, column),
        PlanNode::Sort(sort) => projected_count_preserving_column(plan, sort.input, column),
        _ => false,
    }
}

fn source_reader_predicate_filter_is_elidable(plan: &QueryPlan, filter: &FilterNode) -> bool {
    let Some(node) = plan.node(filter.input) else {
        return false;
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => source.reader_predicate.as_ref() == Some(&filter.predicate),
            _ => false,
        },
        _ => false,
    }
}

fn scalar_aggregate_value_expr_supported(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) => true,
        Expr::Alias { expr, .. } => scalar_aggregate_value_expr_supported(expr),
        _ => false,
    }
}

fn materialization_family_aggregate_exprs(family: &[AggregateMaterializationSignature]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut aggregate_exprs = Vec::new();
    for signature in family {
        for aggregate_expr in &signature.aggregate_exprs {
            if seen.insert(aggregate_expr.clone()) {
                aggregate_exprs.push(aggregate_expr.clone());
            }
        }
    }
    aggregate_exprs
}

fn conjunction(mut predicates: Vec<Expr>) -> Option<Expr> {
    let first = predicates.pop()?;
    Some(predicates.into_iter().fold(first, |right, left| Expr::binary(left, BinaryOp::And, right)))
}

fn disjunction(mut predicates: Vec<Expr>) -> Option<Expr> {
    let first = predicates.pop()?;
    Some(predicates.into_iter().fold(first, |right, left| Expr::binary(left, BinaryOp::Or, right)))
}

fn predicate_references_only_columns(predicate: &Expr, allowed_columns: &BTreeSet<String>) -> bool {
    let mut columns = BTreeSet::new();
    collect_expr_columns(predicate, &mut columns);
    columns.iter().all(|column| allowed_columns.contains(column))
}

fn predicate_references_only_source_filterable_columns(
    predicate: &Expr,
    signature: &AggregateMaterializationSignature,
    source_fingerprint: &str,
) -> bool {
    signature
        .post_aggregate_filter_columns_by_source
        .get(source_fingerprint)
        .is_some_and(|allowed_columns| predicate_references_only_columns(predicate, allowed_columns))
}

fn collect_expr_columns(expr: &Expr, columns: &mut BTreeSet<String>) {
    match expr {
        Expr::Column(column) => {
            columns.insert(column.name().to_string());
        },
        Expr::Literal(_) => {},
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Trim { expr, .. }
        | Expr::LTrim { expr, .. }
        | Expr::RTrim { expr, .. }
        | Expr::StringContains { expr, .. }
        | Expr::StartsWith { expr, .. }
        | Expr::EndsWith { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. } => collect_expr_columns(expr, columns),
        Expr::RegexpReplace { expr, .. } => collect_expr_columns(expr, columns),
        Expr::Concat(exprs) => {
            for expr in exprs {
                collect_expr_columns(expr, columns);
            }
        },
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_expr_columns(left, columns);
            collect_expr_columns(right, columns);
        },
        Expr::InList { expr, list, .. } => {
            collect_expr_columns(expr, columns);
            for expr in list {
                collect_expr_columns(expr, columns);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_expr_columns(&branch.when, columns);
                collect_expr_columns(&branch.then, columns);
            }
            collect_expr_columns(else_expr, columns);
        },
    }
}

fn collect_expr_column_names(expr: &Expr, columns: &mut BTreeSet<String>) {
    collect_expr_columns(expr, columns);
}

fn collect_join_expr_columns(expr: &Expr, left: &mut BTreeSet<String>, right: &mut BTreeSet<String>) {
    match expr {
        Expr::Column(column) => match column.qualifier() {
            Some(ColumnQualifier::Input { index: 0 }) => {
                left.insert(column.name().to_string());
            },
            Some(ColumnQualifier::Input { index: 1 }) => {
                right.insert(column.name().to_string());
            },
            _ => {},
        },
        Expr::Literal(_) => {},
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::Lower(expr)
        | Expr::Upper(expr)
        | Expr::Trim { expr, .. }
        | Expr::LTrim { expr, .. }
        | Expr::RTrim { expr, .. }
        | Expr::StringContains { expr, .. }
        | Expr::StartsWith { expr, .. }
        | Expr::EndsWith { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::Abs(expr)
        | Expr::Sqrt { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. }
        | Expr::RegexpReplace { expr, .. } => collect_join_expr_columns(expr, left, right),
        Expr::Concat(exprs) => {
            for expr in exprs {
                collect_join_expr_columns(expr, left, right);
            }
        },
        Expr::Binary { left: lhs, right: rhs, .. }
        | Expr::IsDistinctFrom { left: lhs, right: rhs }
        | Expr::IsNotDistinctFrom { left: lhs, right: rhs } => {
            collect_join_expr_columns(lhs, left, right);
            collect_join_expr_columns(rhs, left, right);
        },
        Expr::InList { expr, list, .. } => {
            collect_join_expr_columns(expr, left, right);
            for expr in list {
                collect_join_expr_columns(expr, left, right);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_join_expr_columns(&branch.when, left, right);
                collect_join_expr_columns(&branch.then, left, right);
            }
            collect_join_expr_columns(else_expr, left, right);
        },
    }
}

fn field_with_name<'a>(schema: &'a Arc<Schema>, name: &str) -> Option<&'a Field> {
    schema.fields().iter().find(|field| field.name() == name).map(|field| field.as_ref())
}
