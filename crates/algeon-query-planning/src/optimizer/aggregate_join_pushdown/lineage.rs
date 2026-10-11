//! Lineage from the aggregate's group keys and measures down to one join
//! input.
//!
//! Projections between the aggregate and the join are inlined so every
//! expression is stated over the join's output columns, then each column is
//! attributed to a side through the join's output projection. A measure that
//! reads both sides, or a column with no unique join output, is a rejection
//! with the reason that names it.

use std::collections::BTreeMap;

use super::{
    AggregateExpr, AggregateFunction, AggregateJoinPushdownRejectedReason, AggregateNode, CaseWhen, ColumnRef, Expr,
    JoinKey, JoinNode, JoinSide, JoinType, PreAggregateColumn, ProjectionNode, simple_projection_column,
    unique_join_output_for_name,
};

/// One group key traced to a join input.
#[derive(Clone)]
pub(in crate::optimizer::aggregate_join_pushdown) struct LineageColumn {
    pub(in crate::optimizer::aggregate_join_pushdown) side: JoinSide,
    /// Column name on the `side` input.
    pub(in crate::optimizer::aggregate_join_pushdown) source_name: String,
    /// Name the aggregate's output uses for it.
    pub(in crate::optimizer::aggregate_join_pushdown) output_name: String,
}

/// Lineage entry for one aggregate measure expression after the projection
/// chain between the aggregate and the join has been inlined.
#[derive(Clone)]
pub(in crate::optimizer::aggregate_join_pushdown) struct LineageMeasureColumn {
    pub(in crate::optimizer::aggregate_join_pushdown) side: JoinSide,
    /// Aggregate measure expression rewritten against the join's `side`-input
    /// column names. This is the expression the pre-aggregate evaluates.
    pub(in crate::optimizer::aggregate_join_pushdown) side_input_aggregate: AggregateExpr,
}

pub(in crate::optimizer::aggregate_join_pushdown) struct AggregateJoinLineage {
    pub(in crate::optimizer::aggregate_join_pushdown) group_columns: Vec<LineageColumn>,
    pub(in crate::optimizer::aggregate_join_pushdown) measure_columns: Vec<LineageMeasureColumn>,
}

impl AggregateJoinLineage {
    pub(in crate::optimizer::aggregate_join_pushdown) fn new(
        aggregate: &AggregateNode,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<Self, AggregateJoinPushdownRejectedReason> {
        let group_columns = aggregate
            .spec
            .group_exprs
            .iter()
            .map(|expr| group_lineage_column(expr, join, projection_chain))
            .collect::<Result<Vec<_>, _>>()?;
        let measure_columns = aggregate
            .spec
            .aggr_exprs
            .iter()
            .map(|expr| measure_lineage_column(expr, join, projection_chain))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { group_columns, measure_columns })
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn single_group_side(
        &self,
    ) -> Result<JoinSide, AggregateJoinPushdownRejectedReason> {
        single_side(self.group_columns.iter().map(|column| column.side))
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn single_measure_side(
        &self,
    ) -> Result<JoinSide, AggregateJoinPushdownRejectedReason> {
        single_side(self.measure_columns.iter().map(|column| column.side))
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn group_columns_for_side(
        &self,
        side: JoinSide,
    ) -> Vec<PreAggregateColumn> {
        self.group_columns
            .iter()
            .filter(|column| column.side == side)
            .map(|column| PreAggregateColumn {
                source_name: column.source_name.clone(),
                output_name: column.output_name.clone(),
            })
            .collect()
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) fn single_side(
    mut sides: impl Iterator<Item = JoinSide>,
) -> Result<JoinSide, AggregateJoinPushdownRejectedReason> {
    let first = sides.next().ok_or(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern)?;
    if sides.any(|side| side != first) {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
    }
    Ok(first)
}

pub(in crate::optimizer::aggregate_join_pushdown) fn group_lineage_column(
    expr: &Expr,
    join: &JoinNode,
    chain: &[ProjectionNode],
) -> Result<LineageColumn, AggregateJoinPushdownRejectedReason> {
    // Group keys must surface as plain column references at the aggregate
    // boundary. The aggregate's view-name comes from the original group expr;
    // the inlined column resolves which join output column it reads.
    let aggregate_view_column =
        simple_projection_column(expr).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
    let aggregate_output_name = aggregate_view_column.name().to_string();
    let inlined = inline_expr_through_projection_chain(expr.clone(), chain)
        .map_err(|_| AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
    let join_output_column =
        simple_projection_column(&inlined).ok_or(AggregateJoinPushdownRejectedReason::UnsupportedGroupExpression)?;
    let output = unique_join_output_for_name(join, join_output_column.name())
        .or_else(|| unique_join_output_for_name(join, join_output_column.display_name()))
        .ok_or(AggregateJoinPushdownRejectedReason::MissingJoinOutputLineage)?;
    Ok(LineageColumn {
        side: output.side,
        source_name: output.column.name().to_string(),
        output_name: aggregate_output_name,
    })
}

pub(in crate::optimizer::aggregate_join_pushdown) fn measure_lineage_column(
    expr: &AggregateExpr,
    join: &JoinNode,
    chain: &[ProjectionNode],
) -> Result<LineageMeasureColumn, AggregateJoinPushdownRejectedReason> {
    if !matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Count | AggregateFunction::Avg)
        || expr.distinct
        || expr.filter.is_some()
    {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let inlined = inline_expr_through_projection_chain(expr.expr.clone(), chain)
        .map_err(|_| AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression)?;
    let (side, side_input_expr) = resolve_expr_side_and_remap_to_input(&inlined, join)?;
    Ok(LineageMeasureColumn {
        side,
        side_input_aggregate: AggregateExpr {
            function: expr.function,
            expr: side_input_expr,
            alias: expr.alias.clone(),
            distinct: expr.distinct,
            filter: expr.filter.clone(),
        },
    })
}

/// Inlines an expression through a chain of projection nodes. `chain[0]` is the
/// outermost projection (closest to the aggregate) and `chain[last]` is the
/// innermost (closest to the join). After inlining, the returned expression is
/// in terms of the join's output column names.
pub(in crate::optimizer::aggregate_join_pushdown) fn inline_expr_through_projection_chain(
    expr: Expr,
    chain: &[ProjectionNode],
) -> Result<Expr, AggregateJoinPushdownRejectedReason> {
    let mut current = expr;
    for projection in chain {
        current = substitute_columns_through_projection(current, projection)?;
    }
    Ok(current)
}

pub(in crate::optimizer::aggregate_join_pushdown) fn substitute_columns_through_projection(
    expr: Expr,
    projection: &ProjectionNode,
) -> Result<Expr, AggregateJoinPushdownRejectedReason> {
    let mut map: BTreeMap<&str, &Expr> = BTreeMap::new();
    for output in &projection.outputs {
        if map.insert(output.field.name(), &output.expr).is_some() {
            // Two outputs share the same name; ambiguous lineage.
            return Err(AggregateJoinPushdownRejectedReason::MissingJoinOutputLineage);
        }
    }
    Ok(substitute_unqualified_columns(expr, &|name| map.get(name).map(|expr| peel_outer_alias((**expr).clone()))))
}

pub(in crate::optimizer::aggregate_join_pushdown) fn peel_outer_alias(expr: Expr) -> Expr {
    match expr {
        Expr::Alias { expr, .. } => peel_outer_alias(*expr),
        other => other,
    }
}

/// Resolves which side of `join` an expression draws its columns from and
/// rewrites the expression in terms of that side's input column names.
///
/// Returns [`AggregateJoinPushdownRejectedReason::UnsupportedSidePattern`] when
/// the expression mixes columns from both sides of the join.
pub(in crate::optimizer::aggregate_join_pushdown) fn resolve_expr_side_and_remap_to_input(
    expr: &Expr,
    join: &JoinNode,
) -> Result<(JoinSide, Expr), AggregateJoinPushdownRejectedReason> {
    let mut names = Vec::new();
    collect_unqualified_column_names(expr, &mut names);
    if names.is_empty() {
        return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
    }
    let mut resolved_side: Option<JoinSide> = None;
    let mut name_to_input: BTreeMap<String, String> = BTreeMap::new();
    for name in &names {
        let output = unique_join_output_for_name(join, name.as_str())
            .ok_or(AggregateJoinPushdownRejectedReason::MissingJoinOutputLineage)?;
        if let Some(existing) = resolved_side
            && existing != output.side
        {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
        }
        resolved_side = Some(output.side);
        name_to_input.insert(name.clone(), output.column.name().to_string());
    }
    let side = resolved_side.expect("at least one column present");
    let remapped = substitute_unqualified_columns(expr.clone(), &|name| {
        name_to_input.get(name).map(|input_name| Expr::column(input_name.clone()))
    });
    Ok((side, remapped))
}

pub(in crate::optimizer::aggregate_join_pushdown) fn collect_unqualified_column_names(
    expr: &Expr,
    out: &mut Vec<String>,
) {
    match expr {
        Expr::Column(column) => {
            if column.qualifier().is_none() && !out.iter().any(|n| n == column.name()) {
                out.push(column.name().to_string());
            }
        },
        Expr::Literal(_) => {},
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::Trim { expr: inner, .. }
        | Expr::LTrim { expr: inner, .. }
        | Expr::RTrim { expr: inner, .. }
        | Expr::StringContains { expr: inner, .. }
        | Expr::StartsWith { expr: inner, .. }
        | Expr::EndsWith { expr: inner, .. }
        | Expr::Abs(inner) => collect_unqualified_column_names(inner, out),
        Expr::Cast { expr, .. }
        | Expr::TryCast { expr, .. }
        | Expr::Like { expr, .. }
        | Expr::Substring { expr, .. }
        | Expr::StringLength { expr, .. }
        | Expr::RegexpReplace { expr, .. }
        | Expr::UnaryMath { expr, .. }
        | Expr::Round { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::Alias { expr, .. } => collect_unqualified_column_names(expr, out),
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_unqualified_column_names(left, out);
            collect_unqualified_column_names(right, out);
        },
        Expr::Concat(parts) => {
            for part in parts {
                collect_unqualified_column_names(part, out);
            }
        },
        Expr::InList { expr, list, .. } => {
            collect_unqualified_column_names(expr, out);
            for item in list {
                collect_unqualified_column_names(item, out);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_unqualified_column_names(&branch.when, out);
                collect_unqualified_column_names(&branch.then, out);
            }
            collect_unqualified_column_names(else_expr, out);
        },
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) fn substitute_unqualified_columns(
    expr: Expr,
    map: &dyn Fn(&str) -> Option<Expr>,
) -> Expr {
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
        Expr::StringLength { expr, output_type } => {
            Expr::StringLength { expr: Box::new(substitute_unqualified_columns(*expr, map)), output_type }
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

/// Whether rows from `side` can be null-extended by `join_type`, which makes
/// every pre-aggregated state column from that side nullable after the join.
pub(in crate::optimizer::aggregate_join_pushdown) fn join_side_introduces_nulls(
    join_type: &JoinType,
    side: JoinSide,
) -> bool {
    matches!((join_type, side), (JoinType::Left, JoinSide::Right) | (JoinType::Full, JoinSide::Left | JoinSide::Right))
}

/// The join key with the pre-aggregated side's column renamed to the group
/// output that now carries it; the other side's reference is unchanged.
pub(in crate::optimizer::aggregate_join_pushdown) fn rewritten_join_key_after_preagg(
    key: &JoinKey,
    preagg_side: JoinSide,
    group_columns: &[PreAggregateColumn],
) -> JoinKey {
    let source = join_key_for_side(key, preagg_side);
    let output_name = &group_columns
        .iter()
        .find(|column| column.source_name == source.name())
        .expect("every preaggregated join key has a retained group column")
        .output_name;
    match preagg_side {
        JoinSide::Left => JoinKey::new(
            ColumnRef::input(0, output_name, key.left.display_name()),
            ColumnRef::input(1, key.right.name(), key.right.display_name()),
        ),
        JoinSide::Right => JoinKey::new(
            ColumnRef::input(0, key.left.name(), key.left.display_name()),
            ColumnRef::input(1, output_name, key.right.display_name()),
        ),
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) fn join_key_for_side(key: &JoinKey, side: JoinSide) -> &ColumnRef {
    match side {
        JoinSide::Left => &key.left,
        JoinSide::Right => &key.right,
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

pub(in crate::optimizer::aggregate_join_pushdown) const fn opposite_side(side: JoinSide) -> JoinSide {
    match side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    }
}
