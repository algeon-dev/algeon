//! SQL AST recursion guard for nested coverage validation.

use std::ops::ControlFlow;

use datafusion::sql::sqlparser::ast::{
    AccessExpr, Array as SqlArray, Assignment, Expr as SqlExpr, ExprWithAlias, ExprWithAliasAndOrderBy, Fetch,
    Function as SqlFunction, FunctionArg, FunctionArgExpr, FunctionArgumentClause, FunctionArguments, GroupByExpr,
    Join as SqlJoin, JoinConstraint, JoinOperator, LimitClause, Map as SqlMap, ObjectName, ObjectNamePart, OrderByExpr,
    OrderByKind, PipeOperator, PivotValueSource, Query as SqlQuery, Select as SqlSelect, SelectItem, SetExpr,
    Subscript, TableFactor, TableSample, TableWithJoins, Visit, Visitor, WindowSpec, WindowType,
};

use super::NEXUS_EXPLAIN_COVERAGE_FUNCTION;

pub(super) fn query_mentions_coverage_table_function(query: &SqlQuery) -> bool {
    query_ast_mentions_coverage_table_function(query)
        || query
            .with
            .as_ref()
            .is_some_and(|with| with.cte_tables.iter().any(|cte| query_mentions_coverage_table_function(&cte.query)))
        || set_expr_mentions_coverage_table_function(&query.body)
        || query.order_by.as_ref().is_some_and(order_by_mentions_coverage_table_function)
        || query.limit_clause.as_ref().is_some_and(limit_clause_mentions_coverage_table_function)
        || query.fetch.as_ref().is_some_and(fetch_mentions_coverage_table_function)
        || query.pipe_operators.iter().any(pipe_operator_mentions_coverage_table_function)
}

fn query_ast_mentions_coverage_table_function(query: &SqlQuery) -> bool {
    matches!(query.visit(&mut CoverageTableFunctionVisitor), ControlFlow::Break(()))
}

struct CoverageTableFunctionVisitor;

impl Visitor for CoverageTableFunctionVisitor {
    type Break = ();

    fn pre_visit_table_factor(&mut self, table_factor: &TableFactor) -> ControlFlow<Self::Break> {
        let mentions = match table_factor {
            TableFactor::Table { name, args: Some(_), .. } | TableFactor::Function { name, .. } => {
                object_name_is_coverage_table_function(name)
            },
            _ => false,
        };
        if mentions { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    }

    fn pre_visit_expr(&mut self, expr: &SqlExpr) -> ControlFlow<Self::Break> {
        match expr {
            SqlExpr::Function(function) if object_name_is_coverage_table_function(&function.name) => {
                ControlFlow::Break(())
            },
            _ => ControlFlow::Continue(()),
        }
    }
}

fn set_expr_mentions_coverage_table_function(expr: &SetExpr) -> bool {
    match expr {
        SetExpr::Select(select) => select_mentions_coverage_table_function(select),
        SetExpr::Query(query) => query_mentions_coverage_table_function(query),
        SetExpr::SetOperation { left, right, .. } => {
            set_expr_mentions_coverage_table_function(left) || set_expr_mentions_coverage_table_function(right)
        },
        SetExpr::Values(_)
        | SetExpr::Insert(_)
        | SetExpr::Update(_)
        | SetExpr::Delete(_)
        | SetExpr::Merge(_)
        | SetExpr::Table(_) => false,
    }
}

fn select_mentions_coverage_table_function(select: &SqlSelect) -> bool {
    select.projection.iter().any(select_item_mentions_coverage_table_function)
        || select.from.iter().any(table_with_joins_mentions_coverage_table_function)
        || select.lateral_views.iter().any(|view| expr_mentions_coverage_table_function(&view.lateral_view))
        || select.prewhere.as_ref().is_some_and(expr_mentions_coverage_table_function)
        || select.selection.as_ref().is_some_and(expr_mentions_coverage_table_function)
        || group_by_mentions_coverage_table_function(&select.group_by)
        || select.cluster_by.iter().any(expr_mentions_coverage_table_function)
        || select.distribute_by.iter().any(expr_mentions_coverage_table_function)
        || select.sort_by.iter().any(order_by_expr_mentions_coverage_table_function)
        || select.having.as_ref().is_some_and(expr_mentions_coverage_table_function)
        || select.named_window.iter().any(named_window_definition_mentions_coverage_table_function)
        || select.qualify.as_ref().is_some_and(expr_mentions_coverage_table_function)
}

fn select_item_mentions_coverage_table_function(item: &SelectItem) -> bool {
    match item {
        SelectItem::UnnamedExpr(expr)
        | SelectItem::ExprWithAlias { expr, .. }
        | SelectItem::ExprWithAliases { expr, .. } => expr_mentions_coverage_table_function(expr),
        SelectItem::QualifiedWildcard(kind, _) => match kind {
            datafusion::sql::sqlparser::ast::SelectItemQualifiedWildcardKind::Expr(expr) => {
                expr_mentions_coverage_table_function(expr)
            },
            datafusion::sql::sqlparser::ast::SelectItemQualifiedWildcardKind::ObjectName(_) => false,
        },
        SelectItem::Wildcard(_) => false,
    }
}

fn table_with_joins_mentions_coverage_table_function(table: &TableWithJoins) -> bool {
    table_factor_mentions_coverage_table_function(&table.relation)
        || table.joins.iter().any(|join| {
            table_factor_mentions_coverage_table_function(&join.relation)
                || join_operator_mentions_coverage_table_function(&join.join_operator)
        })
}

fn table_factor_mentions_coverage_table_function(factor: &TableFactor) -> bool {
    match factor {
        TableFactor::Table { name, args, .. } => args.is_some() && object_name_is_coverage_table_function(name),
        TableFactor::Function { name, .. } => object_name_is_coverage_table_function(name),
        TableFactor::TableFunction { expr, .. } => expr_mentions_coverage_table_function(expr),
        TableFactor::Derived { subquery, .. } => query_mentions_coverage_table_function(subquery),
        TableFactor::UNNEST { array_exprs, .. } => array_exprs.iter().any(expr_mentions_coverage_table_function),
        TableFactor::NestedJoin { table_with_joins, .. } => {
            table_with_joins_mentions_coverage_table_function(table_with_joins)
        },
        TableFactor::Pivot { table, .. } | TableFactor::Unpivot { table, .. } => {
            table_factor_mentions_coverage_table_function(table)
        },
        _ => false,
    }
}

fn join_operator_mentions_coverage_table_function(operator: &JoinOperator) -> bool {
    match operator {
        JoinOperator::Join(constraint)
        | JoinOperator::Inner(constraint)
        | JoinOperator::Left(constraint)
        | JoinOperator::LeftOuter(constraint)
        | JoinOperator::Right(constraint)
        | JoinOperator::RightOuter(constraint)
        | JoinOperator::FullOuter(constraint)
        | JoinOperator::CrossJoin(constraint)
        | JoinOperator::Semi(constraint)
        | JoinOperator::LeftSemi(constraint)
        | JoinOperator::RightSemi(constraint)
        | JoinOperator::Anti(constraint)
        | JoinOperator::LeftAnti(constraint)
        | JoinOperator::RightAnti(constraint)
        | JoinOperator::StraightJoin(constraint) => join_constraint_mentions_coverage_table_function(constraint),
        JoinOperator::AsOf { match_condition, constraint } => {
            expr_mentions_coverage_table_function(match_condition)
                || join_constraint_mentions_coverage_table_function(constraint)
        },
        JoinOperator::CrossApply
        | JoinOperator::OuterApply
        | JoinOperator::ArrayJoin
        | JoinOperator::LeftArrayJoin
        | JoinOperator::InnerArrayJoin => false,
    }
}

fn pipe_operator_mentions_coverage_table_function(operator: &PipeOperator) -> bool {
    match operator {
        PipeOperator::Limit { expr, offset } => {
            expr_mentions_coverage_table_function(expr)
                || offset.as_ref().is_some_and(expr_mentions_coverage_table_function)
        },
        PipeOperator::Where { expr } => expr_mentions_coverage_table_function(expr),
        PipeOperator::OrderBy { exprs } => exprs.iter().any(order_by_expr_mentions_coverage_table_function),
        PipeOperator::Select { exprs } | PipeOperator::Extend { exprs } => {
            exprs.iter().any(select_item_mentions_coverage_table_function)
        },
        PipeOperator::Set { assignments } => assignments.iter().any(assignment_mentions_coverage_table_function),
        PipeOperator::Aggregate { full_table_exprs, group_by_expr } => {
            full_table_exprs.iter().any(expr_with_alias_and_order_by_mentions_coverage_table_function)
                || group_by_expr.iter().any(expr_with_alias_and_order_by_mentions_coverage_table_function)
        },
        PipeOperator::TableSample { sample } => table_sample_mentions_coverage_table_function(sample),
        PipeOperator::Union { queries, .. }
        | PipeOperator::Intersect { queries, .. }
        | PipeOperator::Except { queries, .. } => queries.iter().any(query_mentions_coverage_table_function),
        PipeOperator::Call { function, .. } => function_mentions_coverage_table_function(function),
        PipeOperator::Pivot { aggregate_functions, value_source, .. } => {
            aggregate_functions.iter().any(expr_with_alias_mentions_coverage_table_function)
                || pivot_value_source_mentions_coverage_table_function(value_source)
        },
        PipeOperator::Join(join) => join_mentions_coverage_table_function(join),
        PipeOperator::Drop { .. }
        | PipeOperator::As { .. }
        | PipeOperator::Rename { .. }
        | PipeOperator::Unpivot { .. } => false,
    }
}

fn assignment_mentions_coverage_table_function(assignment: &Assignment) -> bool {
    expr_mentions_coverage_table_function(&assignment.value)
}

fn expr_with_alias_mentions_coverage_table_function(expr: &ExprWithAlias) -> bool {
    expr_mentions_coverage_table_function(&expr.expr)
}

fn expr_with_alias_and_order_by_mentions_coverage_table_function(expr: &ExprWithAliasAndOrderBy) -> bool {
    expr_with_alias_mentions_coverage_table_function(&expr.expr)
}

fn table_sample_mentions_coverage_table_function(sample: &TableSample) -> bool {
    sample.quantity.as_ref().is_some_and(|quantity| expr_mentions_coverage_table_function(&quantity.value))
        || sample
            .bucket
            .as_ref()
            .and_then(|bucket| bucket.on.as_ref())
            .is_some_and(expr_mentions_coverage_table_function)
        || sample.offset.as_ref().is_some_and(expr_mentions_coverage_table_function)
}

fn pivot_value_source_mentions_coverage_table_function(value_source: &PivotValueSource) -> bool {
    match value_source {
        PivotValueSource::List(values) => values.iter().any(expr_with_alias_mentions_coverage_table_function),
        PivotValueSource::Any(order_by) => order_by.iter().any(order_by_expr_mentions_coverage_table_function),
        PivotValueSource::Subquery(query) => query_mentions_coverage_table_function(query),
    }
}

fn join_mentions_coverage_table_function(join: &SqlJoin) -> bool {
    table_factor_mentions_coverage_table_function(&join.relation)
        || join_operator_mentions_coverage_table_function(&join.join_operator)
}

fn join_constraint_mentions_coverage_table_function(constraint: &JoinConstraint) -> bool {
    match constraint {
        JoinConstraint::On(expr) => expr_mentions_coverage_table_function(expr),
        JoinConstraint::Using(_) | JoinConstraint::Natural | JoinConstraint::None => false,
    }
}

fn group_by_mentions_coverage_table_function(group_by: &GroupByExpr) -> bool {
    match group_by {
        GroupByExpr::All(_) => false,
        GroupByExpr::Expressions(exprs, _) => exprs.iter().any(expr_mentions_coverage_table_function),
    }
}

fn order_by_mentions_coverage_table_function(order_by: &datafusion::sql::sqlparser::ast::OrderBy) -> bool {
    match &order_by.kind {
        OrderByKind::All(_) => false,
        OrderByKind::Expressions(exprs) => exprs.iter().any(order_by_expr_mentions_coverage_table_function),
    }
}

fn order_by_expr_mentions_coverage_table_function(expr: &OrderByExpr) -> bool {
    expr_mentions_coverage_table_function(&expr.expr)
}

fn limit_clause_mentions_coverage_table_function(limit: &LimitClause) -> bool {
    match limit {
        LimitClause::LimitOffset { limit, offset, limit_by } => {
            limit.as_ref().is_some_and(expr_mentions_coverage_table_function)
                || offset.as_ref().is_some_and(|offset| expr_mentions_coverage_table_function(&offset.value))
                || limit_by.iter().any(expr_mentions_coverage_table_function)
        },
        LimitClause::OffsetCommaLimit { offset, limit } => {
            expr_mentions_coverage_table_function(offset) || expr_mentions_coverage_table_function(limit)
        },
    }
}

fn fetch_mentions_coverage_table_function(fetch: &Fetch) -> bool {
    fetch.quantity.as_ref().is_some_and(expr_mentions_coverage_table_function)
}

fn expr_mentions_coverage_table_function(expr: &SqlExpr) -> bool {
    match expr {
        SqlExpr::InSubquery { expr, subquery, .. } => {
            expr_mentions_coverage_table_function(expr) || query_mentions_coverage_table_function(subquery)
        },
        SqlExpr::Exists { subquery, .. } | SqlExpr::Subquery(subquery) => {
            query_mentions_coverage_table_function(subquery)
        },
        SqlExpr::AnyOp { left, right, .. } | SqlExpr::AllOp { left, right, .. } => {
            expr_mentions_coverage_table_function(left) || expr_mentions_coverage_table_function(right)
        },
        SqlExpr::CompoundFieldAccess { root, access_chain } => {
            expr_mentions_coverage_table_function(root)
                || access_chain.iter().any(access_expr_mentions_coverage_table_function)
        },
        SqlExpr::BinaryOp { left, right, .. }
        | SqlExpr::IsDistinctFrom(left, right)
        | SqlExpr::IsNotDistinctFrom(left, right)
        | SqlExpr::AtTimeZone { timestamp: left, time_zone: right }
        | SqlExpr::Position { expr: left, r#in: right } => {
            expr_mentions_coverage_table_function(left) || expr_mentions_coverage_table_function(right)
        },
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsFalse(expr)
        | SqlExpr::IsNotFalse(expr)
        | SqlExpr::IsTrue(expr)
        | SqlExpr::IsNotTrue(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::IsUnknown(expr)
        | SqlExpr::IsNotUnknown(expr)
        | SqlExpr::IsNormalized { expr, .. }
        | SqlExpr::OuterJoin(expr)
        | SqlExpr::Prior(expr)
        | SqlExpr::Prefixed { value: expr, .. }
        | SqlExpr::Named { expr, .. }
        | SqlExpr::Collate { expr, .. }
        | SqlExpr::Cast { expr, .. }
        | SqlExpr::Extract { expr, .. }
        | SqlExpr::Ceil { expr, .. }
        | SqlExpr::Floor { expr, .. } => expr_mentions_coverage_table_function(expr),
        SqlExpr::Convert { expr, styles, .. } => {
            expr_mentions_coverage_table_function(expr) || styles.iter().any(expr_mentions_coverage_table_function)
        },
        SqlExpr::JsonAccess { value, .. } => expr_mentions_coverage_table_function(value),
        SqlExpr::InList { expr, list, .. } => {
            expr_mentions_coverage_table_function(expr) || list.iter().any(expr_mentions_coverage_table_function)
        },
        SqlExpr::InUnnest { expr, array_expr, .. } => {
            expr_mentions_coverage_table_function(expr) || expr_mentions_coverage_table_function(array_expr)
        },
        SqlExpr::Between { expr, low, high, .. } => {
            expr_mentions_coverage_table_function(expr)
                || expr_mentions_coverage_table_function(low)
                || expr_mentions_coverage_table_function(high)
        },
        SqlExpr::Like { expr, pattern, .. }
        | SqlExpr::ILike { expr, pattern, .. }
        | SqlExpr::SimilarTo { expr, pattern, .. }
        | SqlExpr::RLike { expr, pattern, .. } => {
            expr_mentions_coverage_table_function(expr) || expr_mentions_coverage_table_function(pattern)
        },
        SqlExpr::Substring { expr, substring_from, substring_for, .. } => {
            expr_mentions_coverage_table_function(expr)
                || substring_from.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
                || substring_for.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        },
        SqlExpr::Trim { expr, trim_what, trim_characters, .. } => {
            expr_mentions_coverage_table_function(expr)
                || trim_what.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
                || trim_characters.as_ref().is_some_and(|exprs| exprs.iter().any(expr_mentions_coverage_table_function))
        },
        SqlExpr::Overlay { expr, overlay_what, overlay_from, overlay_for } => {
            expr_mentions_coverage_table_function(expr)
                || expr_mentions_coverage_table_function(overlay_what)
                || expr_mentions_coverage_table_function(overlay_from)
                || overlay_for.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        },
        SqlExpr::Function(function) => function_mentions_coverage_table_function(function),
        SqlExpr::Case { operand, conditions, else_result, .. } => {
            operand.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
                || conditions.iter().any(|case| {
                    expr_mentions_coverage_table_function(&case.condition)
                        || expr_mentions_coverage_table_function(&case.result)
                })
                || else_result.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        },
        SqlExpr::GroupingSets(sets) | SqlExpr::Cube(sets) | SqlExpr::Rollup(sets) => {
            sets.iter().flat_map(|set| set.iter()).any(expr_mentions_coverage_table_function)
        },
        SqlExpr::Tuple(exprs) => exprs.iter().any(expr_mentions_coverage_table_function),
        SqlExpr::Struct { values, .. } => values.iter().any(expr_mentions_coverage_table_function),
        SqlExpr::Dictionary(fields) => fields.iter().any(|field| expr_mentions_coverage_table_function(&field.value)),
        SqlExpr::Map(map) => map_mentions_coverage_table_function(map),
        SqlExpr::Array(array) => array_mentions_coverage_table_function(array),
        SqlExpr::Interval(interval) => expr_mentions_coverage_table_function(&interval.value),
        SqlExpr::Lambda(lambda) => expr_mentions_coverage_table_function(&lambda.body),
        SqlExpr::MemberOf(member_of) => {
            expr_mentions_coverage_table_function(&member_of.value)
                || expr_mentions_coverage_table_function(&member_of.array)
        },
        _ => false,
    }
}

fn access_expr_mentions_coverage_table_function(access: &AccessExpr) -> bool {
    match access {
        AccessExpr::Dot(expr) => expr_mentions_coverage_table_function(expr),
        AccessExpr::Subscript(subscript) => subscript_mentions_coverage_table_function(subscript),
    }
}

fn subscript_mentions_coverage_table_function(subscript: &Subscript) -> bool {
    match subscript {
        Subscript::Index { index } => expr_mentions_coverage_table_function(index),
        Subscript::Slice { lower_bound, upper_bound, stride } => {
            lower_bound.as_ref().is_some_and(expr_mentions_coverage_table_function)
                || upper_bound.as_ref().is_some_and(expr_mentions_coverage_table_function)
                || stride.as_ref().is_some_and(expr_mentions_coverage_table_function)
        },
    }
}

fn array_mentions_coverage_table_function(array: &SqlArray) -> bool {
    array.elem.iter().any(expr_mentions_coverage_table_function)
}

fn map_mentions_coverage_table_function(map: &SqlMap) -> bool {
    map.entries.iter().any(|entry| {
        expr_mentions_coverage_table_function(&entry.key) || expr_mentions_coverage_table_function(&entry.value)
    })
}

fn function_mentions_coverage_table_function(function: &SqlFunction) -> bool {
    object_name_is_coverage_table_function(&function.name)
        || function_args_mentions_coverage_table_function(&function.parameters)
        || function_args_mentions_coverage_table_function(&function.args)
        || function.filter.as_ref().is_some_and(|expr| expr_mentions_coverage_table_function(expr))
        || function.within_group.iter().any(order_by_expr_mentions_coverage_table_function)
        || function.over.as_ref().is_some_and(window_type_mentions_coverage_table_function)
}

fn function_args_mentions_coverage_table_function(args: &FunctionArguments) -> bool {
    match args {
        FunctionArguments::None => false,
        FunctionArguments::Subquery(query) => query_mentions_coverage_table_function(query),
        FunctionArguments::List(list) => {
            list.args.iter().any(function_arg_mentions_coverage_table_function)
                || list.clauses.iter().any(function_arg_clause_mentions_coverage_table_function)
        },
    }
}

fn function_arg_mentions_coverage_table_function(arg: &FunctionArg) -> bool {
    match arg {
        FunctionArg::Named { arg, .. } => function_arg_expr_mentions_coverage_table_function(arg),
        FunctionArg::ExprNamed { name, arg, .. } => {
            expr_mentions_coverage_table_function(name) || function_arg_expr_mentions_coverage_table_function(arg)
        },
        FunctionArg::Unnamed(arg) => function_arg_expr_mentions_coverage_table_function(arg),
    }
}

fn function_arg_expr_mentions_coverage_table_function(arg: &FunctionArgExpr) -> bool {
    match arg {
        FunctionArgExpr::Expr(expr) => expr_mentions_coverage_table_function(expr),
        FunctionArgExpr::QualifiedWildcard(_) | FunctionArgExpr::Wildcard | FunctionArgExpr::WildcardWithOptions(_) => {
            false
        },
    }
}

fn function_arg_clause_mentions_coverage_table_function(clause: &FunctionArgumentClause) -> bool {
    match clause {
        FunctionArgumentClause::OrderBy(exprs) => exprs.iter().any(order_by_expr_mentions_coverage_table_function),
        FunctionArgumentClause::Limit(expr) => expr_mentions_coverage_table_function(expr),
        FunctionArgumentClause::Having(bound) => expr_mentions_coverage_table_function(&bound.1),
        FunctionArgumentClause::IgnoreOrRespectNulls(_)
        | FunctionArgumentClause::OnOverflow(_)
        | FunctionArgumentClause::Separator(_)
        | FunctionArgumentClause::JsonNullClause(_)
        | FunctionArgumentClause::JsonReturningClause(_) => false,
    }
}

fn window_type_mentions_coverage_table_function(window: &WindowType) -> bool {
    match window {
        WindowType::NamedWindow(_) => false,
        WindowType::WindowSpec(spec) => window_spec_mentions_coverage_table_function(spec),
    }
}

fn named_window_definition_mentions_coverage_table_function(
    definition: &datafusion::sql::sqlparser::ast::NamedWindowDefinition,
) -> bool {
    match &definition.1 {
        datafusion::sql::sqlparser::ast::NamedWindowExpr::NamedWindow(_) => false,
        datafusion::sql::sqlparser::ast::NamedWindowExpr::WindowSpec(spec) => {
            window_spec_mentions_coverage_table_function(spec)
        },
    }
}

fn window_spec_mentions_coverage_table_function(spec: &WindowSpec) -> bool {
    spec.partition_by.iter().any(expr_mentions_coverage_table_function)
        || spec.order_by.iter().any(order_by_expr_mentions_coverage_table_function)
}

fn object_name_is_coverage_table_function(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(ObjectNamePart::as_ident)
        .is_some_and(|ident| ident.value.eq_ignore_ascii_case(NEXUS_EXPLAIN_COVERAGE_FUNCTION))
}
