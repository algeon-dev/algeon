use std::sync::Arc;

use datafusion::{
    common::Result,
    logical_expr::{
        Extension, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    sql::sqlparser::ast::{Expr as SqlExpr, FunctionArg, FunctionArgExpr, TableFactor, Value as SqlValue},
};

use super::{CuvsLogicalNode, CuvsOperation, error::invalid_call};

#[derive(Debug, Default)]
pub struct CuvsRelationPlanner;

pub fn cuvs_relation_planner() -> Arc<dyn RelationPlanner> {
    Arc::new(CuvsRelationPlanner)
}

impl RelationPlanner for CuvsRelationPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> Result<RelationPlanning> {
        let Some(operation) = recognized_operation(&relation, context) else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };

        let TableFactor::Table {
            alias,
            args: Some(function_args),
            with_hints,
            version,
            with_ordinality,
            partitions,
            json_path,
            sample,
            index_hints,
            ..
        } = relation
        else {
            unreachable!("recognized cuVS relation is a table function")
        };
        if function_args.settings.is_some()
            || !with_hints.is_empty()
            || version.is_some()
            || with_ordinality
            || !partitions.is_empty()
            || json_path.is_some()
            || sample.is_some()
            || !index_hints.is_empty()
        {
            return Err(invalid_call(
                operation,
                "unsupported_table_function_modifier",
                "cuVS relation functions accept only positional relation subqueries, options JSON, and a result alias",
            ));
        }

        let descriptor = operation.descriptor();
        let expected_args = descriptor.relation_roles.len() + 1;
        if function_args.args.len() != expected_args {
            return Err(invalid_call(
                operation,
                "invalid_argument_count",
                format!(
                    "{} requires {} parenthesized relation subquery argument(s) followed by one options JSON literal",
                    descriptor.sql_function_name,
                    descriptor.relation_roles.len()
                ),
            ));
        }

        let mut args = function_args.args.into_iter();
        let mut inputs = Vec::with_capacity(descriptor.relation_roles.len());
        for role in descriptor.relation_roles {
            let expr = positional_expression(
                operation,
                args.next().expect("argument count was checked"),
                "relation arguments must be positional subqueries",
            )?;
            inputs.push(plan_relation_subquery(operation, expr, role.as_str(), context)?);
        }
        let options_expr = positional_expression(
            operation,
            args.next().expect("argument count was checked"),
            "options must be a positional JSON string literal",
        )?;
        let options_json = options_literal(operation, options_expr)?;
        let node = CuvsLogicalNode::try_new(operation, inputs, &options_json)?;
        let plan = LogicalPlan::Extension(Extension { node: Arc::new(node) });
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(plan, alias))))
    }
}

fn recognized_operation(relation: &TableFactor, context: &dyn RelationPlannerContext) -> Option<CuvsOperation> {
    let TableFactor::Table { name, args: Some(_), .. } = relation else {
        return None;
    };
    if name.0.len() != 1 {
        return None;
    }
    let ident = name.0.first()?.as_ident()?.clone();
    CuvsOperation::from_sql_function_name(&context.normalize_ident(ident))
}

fn positional_expression(operation: CuvsOperation, argument: FunctionArg, message: &'static str) -> Result<SqlExpr> {
    match argument {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Ok(expr),
        _ => Err(invalid_call(operation, "invalid_argument_shape", message)),
    }
}

fn plan_relation_subquery(
    operation: CuvsOperation,
    expression: SqlExpr,
    role: &'static str,
    context: &mut dyn RelationPlannerContext,
) -> Result<LogicalPlan> {
    match expression {
        SqlExpr::Subquery(subquery) => {
            context.plan(TableFactor::Derived { lateral: false, subquery, alias: None, sample: None })
        },
        _ => Err(invalid_call(
            operation,
            "relation_argument_not_subquery",
            format!("cuVS {role} argument must be a parenthesized relation subquery"),
        )),
    }
}

fn options_literal(operation: CuvsOperation, expression: SqlExpr) -> Result<String> {
    match expression {
        SqlExpr::Value(value) => match value.value {
            SqlValue::SingleQuotedString(value) => Ok(value),
            _ => Err(invalid_call(
                operation,
                "options_not_string_literal",
                "cuVS options must be a non-null single-quoted JSON string literal",
            )),
        },
        _ => Err(invalid_call(
            operation,
            "options_not_string_literal",
            "cuVS options must be a non-null single-quoted JSON string literal",
        )),
    }
}
