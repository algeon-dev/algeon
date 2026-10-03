use std::sync::Arc;

use datafusion::{
    common::{DFSchema, DataFusionError, Result as DataFusionResult},
    logical_expr::{
        Extension, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    sql::sqlparser::ast::{Expr as SqlExpr, TableFactor},
};
use serde_json::{Map, Value};

use super::{
    CuvsLogicalNode, CuvsOperation, GpuOptionKind,
    error::{input_ordering_key_not_projected, invalid_call},
    options::options_from_named_values,
};
use crate::{
    error::Error,
    gpu_functions::{
        bind_named_arguments,
        dense_relation::dense_relation_argument_definitions,
        resolve_input_ordering,
        scalar_arguments::{scalar_json, scalar_literal},
    },
};

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
    ) -> DataFusionResult<RelationPlanning> {
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
        if !with_hints.is_empty()
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
                "cuVS relation functions accept named arguments and a result alias",
            )
            .into());
        }

        let definitions = dense_relation_argument_definitions(
            operation.descriptor().relation_roles.iter().map(|role| role.as_str()),
            operation.descriptor().options,
        );
        let mut arguments =
            bind_named_arguments(operation.descriptor().sql_function_name, function_args, &definitions, |ident| {
                context.normalize_ident(ident)
            })?;
        let mut inputs = Vec::with_capacity(operation.descriptor().relation_roles.len());
        let mut input_orderings = Vec::with_capacity(operation.descriptor().relation_roles.len());
        for role in operation.descriptor().relation_roles {
            let query = arguments
                .take_relation(role.as_str())
                .expect("required cuVS relation argument was checked by the named binder");
            let input = context.plan(TableFactor::Derived {
                lateral: false,
                subquery: query.clone(),
                alias: None,
                sample: None,
            })?;
            input_orderings.push(resolve_input_ordering(&query, &input, context, |key| {
                input_ordering_key_not_projected(operation, role.as_str(), key)
            })?);
            inputs.push(input);
        }

        let mut values = Map::new();
        for option in operation.descriptor().options {
            if matches!(option.kind, GpuOptionKind::RelationBinding) {
                continue;
            }
            if let Some(expression) = arguments.take_value(option.name) {
                values.insert(option.name.to_string(), value_argument(operation, option.name, expression, context)?);
            }
        }
        debug_assert!(arguments.is_empty());
        let options_json = options_from_named_values(operation, &inputs, values)?;
        let node = CuvsLogicalNode::try_new(operation, inputs, &options_json)?.with_input_orderings(input_orderings)?;
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

fn value_argument(
    operation: CuvsOperation,
    name: &'static str,
    expression: SqlExpr,
    context: &mut dyn RelationPlannerContext,
) -> DataFusionResult<Value> {
    let expression = context.sql_expr_to_logical_expr(expression, &DFSchema::empty())?;
    let invalid = || invalid_value(operation, name);
    scalar_literal(&invalid, expression).and_then(|value| scalar_json(&invalid, value)).map_err(DataFusionError::from)
}

fn invalid_value(operation: CuvsOperation, name: &'static str) -> Error {
    invalid_call(operation, "value_argument_not_literal", format!("cuVS argument '{name}' must be a scalar literal"))
}
